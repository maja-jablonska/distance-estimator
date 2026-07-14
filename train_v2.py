#!/usr/bin/env python3
"""
train_v2.py — linear-anchored gated-residual spectrophotometric parallax (driver).

Thin CLI over the spphot package: spphot.data.prepare_sample builds the same
sample/features/zero-point/pixel-masks/A-B split as run_full_gadi.py and
train_nn.py; spphot.v2 holds the JAX model, likelihoods, staged AdamW and
checkpoint I/O (see its docstring for the architecture).

Staged unfreezing (each boundary is an ablation checkpoint):
  stage 1: theta, b only — plain Gaussian warm start on the plx-S/N > warm-snr
           subset (or --init-linear <model.npz> to skip it).
  stage 2: + pi0, eps, scatter head s(x), bilinear phi — full mixture likelihood.
  stage 3: + residual net g (zero-init output -> continuous loss at the switch).

Anchors: --anchors table (parquet/csv) with columns sdss_id, mu, mu_err, group;
each group's stars are reassigned to ONE fold (no cluster-level leakage).

Usage (see train_v2.pbs):
  python train_v2.py --parquet <spectra_zpt.parquet> --allstar <allStar.fits> \
      --pixel-mask-dir <dir> --out v2_results.parquet \
      [--stages 3] [--anchors anchors.parquet] [--init-linear spphot_model.npz]
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

from spphot.data import LABEL_COLS, log, prepare_sample                     # noqa: F401
from spphot.v2 import (N_PHOT, rjce_aks, load_anchors, standardize_fit,     # noqa: F401
                       train_fold, predict_v2, save_checkpoint)
import spphot.eval as E

import jax
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--out", default="v2_results.parquet")
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--pixel-mask-dir", default=None)
    ap.add_argument("--snr-min", type=float, default=100.0)
    ap.add_argument("--bad-frac", type=float, default=0.01)
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    # architecture
    ap.add_argument("--pca-k", type=int, default=128,
                    help="reduced spectral dims for the gate + residual net")
    ap.add_argument("--g-hidden", default="128,128")
    ap.add_argument("--s-hidden", default="128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--gate-tau", type=float, default=2.0,
                    help="hull score above which the residual net switches off")
    ap.add_argument("--gate-width", type=float, default=0.25)
    # staging
    ap.add_argument("--stages", type=int, default=3, choices=[1, 2, 3],
                    help="1=linear backbone, 2=+mixture/pi0/scatter/bilinear, 3=+gated NN")
    ap.add_argument("--epochs1", type=int, default=15)
    ap.add_argument("--epochs2", type=int, default=25)
    ap.add_argument("--epochs3", type=int, default=30)
    ap.add_argument("--warm-snr", type=float, default=5.0,
                    help="stage-1 subset: plx/e_plx above this (warm start only)")
    ap.add_argument("--init-linear", default=None,
                    help="run_full_gadi model .npz to warm-start theta (skips stage 1)")
    # likelihood
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sig-bad", type=float, default=1.0,
                    help="extra std (mas) of the mixture's bad component")
    ap.add_argument("--anchors", default=None,
                    help="parquet/csv: sdss_id, mu, mu_err, group")
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--anchor-floor", type=float, default=0.05,
                    help="per-star systematic floor on anchor mu (mag)")
    # regularization / optimization
    ap.add_argument("--l1-spec", type=float, default=1e-4)
    ap.add_argument("--l1-phi", type=float, default=1e-3)
    ap.add_argument("--lam-g", type=float, default=1e-3,
                    help="pull-to-zero penalty on the residual net output")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="spphot-v2")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    import pandas as pd

    if args.device != "auto":
        # must land before the first jax computation (backend init is lazy)
        os.environ["JAX_PLATFORMS"] = args.device
        os.environ["JAX_PLATFORM_NAME"] = args.device
    log(f"jax devices={jax.devices()}  stages={args.stages}  "
        f"pca_k={args.pca_k}  beta={args.beta}")

    # ---- identical sample as the linear baseline / train_nn ----
    S = prepare_sample(args.parquet, args.allstar, snr_min=args.snr_min,
                       bad_frac=args.bad_frac, batch_rows=args.batch_rows,
                       pixel_mask_dir=args.pixel_mask_dir, seed=args.seed)
    phot = S["phot"].astype(np.float32)
    a_ks = rjce_aks(phot).astype(np.float32)
    feats = np.hstack([phot, a_ks[:, None], S["spec"]]).astype(np.float32, copy=False)
    del S["spec"]
    plx_k, err_k = S["plx"], S["err"]
    samp_k, train_k, ids_k = S["sample"].copy(), S["train"], S["ids"]
    log(f"features: {feats.shape[0]} x {feats.shape[1]} (phot {N_PHOT} + A_rjce 1 + "
        f"spec {feats.shape[1]-N_PHOT-1});  A_Ks(rjce) median "
        f"{np.median(a_ks):.3f}, p95 {np.percentile(a_ks, 95):.3f}")

    anchors = load_anchors(args.anchors, ids_k, samp_k) if args.anchors else None

    A = train_k & (samp_k == "A")
    B = train_k & (samp_k == "B")
    rest = ~train_k

    def anchor_subset(folds):
        if anchors is None:
            return None
        m = np.isin(anchors["fold"], folds)
        return {k: v[m] for k, v in anchors.items()}

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project,
                             name=args.run_name or f"v2-stages{args.stages}",
                             config=vars(args))
        except ImportError:
            log("wandb not installed -> skipping logging")
    cb = ((lambda lbl, st, ep, nll, lr: run.log(
        {f"{lbl}/s{st}_nll": nll, f"{lbl}/lr": lr})) if run else None)

    def fit(mask, folds, label):
        mu, sd = standardize_fit(feats[mask])
        fold_name = folds[0] if len(folds) == 1 else "all"
        params, buf = train_fold(feats, mask, plx_k, err_k, mu, sd,
                                 anchor_subset(folds), fold_name, args, label,
                                 log_cb=cb)
        return params, buf

    log("=== training fold models (staged) ===")
    par_A, buf_A = fit(A, ["A"], "fold-A")
    par_B, buf_B = fit(B, ["B"], "fold-B")
    par_all, buf_all = fit(train_k, ["A", "B"], "all-train")

    def predict_into(triples):
        cols = {k: np.empty(len(plx_k)) for k in ("m", "s", "gate", "d")}
        for mask, params, buf in triples:
            idx = np.where(mask)[0]
            if idx.size:
                m, s, gt, d = predict_v2(params, buf, feats, idx)
                for k, v in zip(("m", "s", "gate", "d"), (m, s, gt, d)):
                    cols[k][idx] = v
        return cols

    # held-out (honest) and in-fold (overfit gauge), as in train_nn
    ho = predict_into([(A, par_B, buf_B), (B, par_A, buf_A),
                       (rest, par_all, buf_all)])
    inf = predict_into([(A, par_A, buf_A), (B, par_B, buf_B)])
    plx_sp, err_sp = ho["m"], ho["s"]

    pd.DataFrame({
        "sdss_id": ids_k,
        "plx": plx_k, "e_plx": err_k,
        "plx_raw": S["plx_raw"], "zeropoint": S["zeropoint"],
        "plx_sp": plx_sp, "err_sp": err_sp,
        "dist_sp_kpc": 1.0 / plx_sp,
        "r_med_photogeo_pc": S["dist_bj"],
        "sample": samp_k, "train": train_k,
        "spec_bad_frac": S["star_bad"],
        "a_ks_rjce": a_ks, "hull_d": ho["d"], "gate": ho["gate"],
    }).to_parquet(args.out, index=False)

    model_out = args.model_out or os.path.splitext(args.out)[0] + "_model.npz"
    save_checkpoint(model_out, par_all, buf_all, {
        "good_pixel_mask": S["good"],
        "label_cols": np.array(LABEL_COLS),
        "args_json": np.array(json.dumps(vars(args))),
    })
    log(f"wrote {args.out} ({len(plx_k)} stars); saved model -> {model_out}")

    # ---- honesty budget: how much work did each tier do? ----
    eps = float(jax.nn.sigmoid(par_all["eps_logit"]))
    pi0 = float(par_all["pi0"])
    phi_l1 = float(jnp.abs(par_all["phi"]).sum())
    gtr = ho["gate"][train_k]
    print(f"\n=== honesty budget (all-train model) ===")
    print(f"  pi0 (residual zero-point)     : {pi0:+.4f} mas")
    print(f"  eps (bad-component fraction)  : {eps:.3f}")
    print(f"  ||phi||_1 (extinction drift)  : {phi_l1:.4f}")
    print(f"  gate: median {np.median(gtr):.3f}, frac<0.5 {np.mean(gtr < 0.5):.3f} "
          f"(train stars outside their own hull should be ~0)")
    print(f"  hull_d: median {np.median(ho['d'][train_k]):.2f}, "
          f"p99 {np.percentile(ho['d'][train_k], 99):.2f}")

    # ---- eval: identical footing to the linear baseline and train_nn ----
    def make_cat(p, e):
        return {"plx_a": plx_k[train_k], "err_a": err_k[train_k],
                "plx_sp": p[train_k], "err_sp": e[train_k],
                "train": np.ones(int(train_k.sum()), bool),
                "sample": samp_k[train_k], "id": ids_k[train_k]}
    cat, cat_in = make_cat(plx_sp, err_sp), make_cat(inf["m"], inf["s"])
    label = f"v2 (stages={args.stages}, beta={args.beta:g})"
    rep = E.evaluate(cat, label=label)
    rep_in = E.evaluate(cat_in, label="in-fold")
    gap = E.overfit_gap(rep_in, rep)
    bins, c = E.calibrate(cat)
    print()
    E.print_report(rep)
    E.print_overfit_gap(rep_in, rep)
    E.print_calibration(bins, c)
    ca = (cat["plx_sp"], cat["plx_a"], cat["err_a"], cat["err_sp"])
    rho = E.sharpness(*ca)
    z = E.zscore(*ca)
    print(f"  sharpness rho {rho:+.3f}; z width {E.robust_scatter(z):.2f}; "
          f"|z|>3 {100*float(np.mean(np.abs(z) > 3)):.2f}%")
    E.print_coverage(E.coverage(*ca))
    for f in ("A", "B"):
        print()
        E.print_report(E.evaluate(cat, fold=f, label=label))

    if run is not None:
        run.log({"scatter": rep["robust_frac_scatter"],
                 "bias": rep["median_frac_resid"],
                 "chi2_robust": rep["chi2_robust_chi2"],
                 "overfit_gap_pp": gap["gap_pp"], "recal_factor": c,
                 "pi0": pi0, "eps": eps, "phi_l1": phi_l1,
                 "gate_median": float(np.median(gtr))})
        run.finish()


if __name__ == "__main__":
    main()
