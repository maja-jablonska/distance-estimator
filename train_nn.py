#!/usr/bin/env python3
"""
train_nn.py — heteroscedastic neural-net spectrophotometric parallax (driver).

Thin CLI over the spphot package: spphot.data.prepare_sample builds the same
sample/features/zero-point/pixel-masks/A-B split as the linear baseline, and
spphot.nn holds the model, losses, training loop and checkpoint I/O. See
NN_MODEL.md for the method, results and how to read the eval report.

Usage (see train_nn.pbs):
  python train_nn.py --parquet <spectra_zpt.parquet> --allstar <allStar.fits> \
        --pixel-mask-dir <dir> --out nn_results.parquet \
        [--hidden 1024,512,256] [--epochs 60] [--batch-size 4096] [--device auto]

  # beta-NLL calibration<->bias sweep (one sample prep, one model per beta):
  python train_nn.py ... --out nn.parquet --beta-sweep 0.2,0.3,0.5,0.7
  #   -> nn_gauss-beta0.2.parquet, ... ; compare bias vs robust chi2 across them
"""
from __future__ import annotations
import os, math, argparse
import numpy as np

from spphot.data import LABEL_COLS, log, prepare_sample                     # noqa: F401
from spphot.datasets import REGISTRY, get_dataset
from spphot.nn import (LOG_SIG_MIN, LOG_SIG_MAX, HetMLP, het_nll,           # noqa: F401
                       studentt_nll, clamp_saturation, standardize_fit,
                       train_fold, predict_nn, save_nn_checkpoint)
import spphot.eval as E

import torch


# ----------------------------------------------------------------------
# optional Weights & Biases logging (offline-friendly)
# ----------------------------------------------------------------------
def _init_wandb(args, tag, config):
    """Start a wandb run for one config, or return None if --wandb is off / wandb
    is missing. Offline mode is controlled by WANDB_MODE=offline in the PBS job
    (compute nodes have no internet); the copyq sync job uploads afterwards."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        log("wandb not installed -> skipping logging")
        return None
    name = f"{args.run_name}-{tag}" if args.run_name else tag
    return wandb.init(project=args.wandb_project, name=name,
                      group=args.run_name or None, config=config, reinit=True)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--out", default="nn_results.parquet")
    ap.add_argument("--model-out", default=None,
                    help="path for the saved torch checkpoint (default: <out>_model.pt)")
    ap.add_argument("--pixel-mask-dir", default=None)
    ap.add_argument("--snr-min", type=float, default=100.0)
    ap.add_argument("--bad-frac", type=float, default=0.01)
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="dr17", choices=sorted(REGISTRY),
                    help="photometry dataset spec (spphot.datasets registry)")
    ap.add_argument("--aux-phot", default=None,
                    help="auxiliary photometry parquet (dataset must declare aux_phot)")
    # NN hyper-parameters
    ap.add_argument("--hidden", default="1024,512,256",
                    help="comma-separated MLP widths")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--loss", default="gauss", choices=["gauss", "studentt"],
                    help="likelihood: gauss (heteroscedastic Gaussian) or studentt "
                         "(fat-tailed, robust to outliers during training)")
    ap.add_argument("--nu", type=float, default=4.0,
                    help="Student-t degrees of freedom (only used with --loss studentt; "
                         "smaller = heavier tails / more robust)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="beta-NLL weight for --loss gauss (Seitzer+22): 0 = plain NLL "
                         "(underfits the mean -> biased), 0.5 = recommended, 1 = MSE-like")
    ap.add_argument("--beta-sweep", default=None,
                    help="comma-separated betas, e.g. '0.2,0.3,0.5,0.7'. Trains one "
                         "gauss model per beta, REUSING the sample prep once; writes "
                         "<out>_gauss-beta<b>.parquet for each. Overrides --loss/--beta.")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (one run per config). On compute "
                         "nodes set WANDB_MODE=offline and sync later from copyq.")
    ap.add_argument("--wandb-project", default="spphot-nn")
    ap.add_argument("--run-name", default=None,
                    help="wandb run name / sweep group; each config appends its tag")
    args = ap.parse_args()
    import pandas as pd

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1)))
    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    log(f"device={device}  hidden={hidden}  epochs={args.epochs}  batch={args.batch_size}")

    # ---- identical sample/features/splits as the linear pipeline. Built ONCE and
    #      reused across every sweep config (this is the expensive step). ----
    spec_ds = get_dataset(args.dataset)
    S = prepare_sample(args.parquet, args.allstar, dataset=spec_ds,
                       aux_phot_path=args.aux_phot, snr_min=args.snr_min,
                       bad_frac=args.bad_frac, batch_rows=args.batch_rows,
                       pixel_mask_dir=args.pixel_mask_dir, seed=args.seed)
    spec = S["spec"]
    n_phot = S["n_phot"]
    # features = [photometry | ln-flux]; float32 for the net (frees the separate spec)
    feats = np.hstack([S["phot"].astype(np.float32), spec]).astype(np.float32, copy=False)
    del spec, S["spec"]
    plx_k, err_k = S["plx"], S["err"]
    samp_k, train_k = S["sample"], S["train"]
    ids_k = S["ids"]
    log(f"features: {feats.shape[0]} stars x {feats.shape[1]} "
        f"(phot {n_phot} + spec {feats.shape[1]-n_phot})")

    A = train_k & (samp_k == "A")
    B = train_k & (samp_k == "B")
    rest = ~train_k

    hp = dict(hidden=hidden, dropout=args.dropout, lr=args.lr,
              weight_decay=args.weight_decay, epochs=args.epochs,
              batch_size=args.batch_size, device=device, seed=args.seed)

    def make_loss(kind, beta, nu):
        """-> (loss_fn, std_factor, tag). std_factor turns the t SCALE into a
        1-sigma std for the output/eval (1.0 for the Gaussian)."""
        if kind == "studentt":
            sf = math.sqrt(nu / (nu - 2.0)) if nu > 2.0 else 1.0
            return (lambda zmu, zls, y, e: studentt_nll(zmu, zls, y, e, nu), sf,
                    f"studentt-nu{nu:g}")
        return (lambda zmu, zls, y, e: het_nll(zmu, zls, y, e, beta=beta), 1.0,
                f"gauss-beta{beta:g}")

    def run_config(loss_fn, std_factor, tag, out_path, model_out, meta):
        run = _init_wandb(args, tag, {**hp, **meta, "std_factor": std_factor,
                                      "snr_min": args.snr_min, "bad_frac": args.bad_frac,
                                      "n_features": int(feats.shape[1]),
                                      "n_train": int(train_k.sum())})
        cb = ((lambda lbl, ep, nll, lr: run.log({f"{lbl}/nll": nll, f"{lbl}/lr": lr}))
              if run is not None else None)

        def fit(train_mask, label):
            mu, sd = standardize_fit(feats[train_mask])
            log(f"  training {label}: {int(train_mask.sum())} stars")
            model = train_fold(feats, train_mask, plx_k, err_k, mu, sd,
                               loss_fn=loss_fn, label=label, log_cb=cb, **hp)
            return model, mu, sd

        log(f"=== config {tag}: training models ===")
        model_A, muA, sdA = fit(A, "fold-A")
        model_B, muB, sdB = fit(B, "fold-B")
        model_all, muAll, sdAll = fit(train_k, "all-train")

        def predict_into(pairs):
            p = np.empty(len(plx_k)); s = np.empty(len(plx_k))
            for mask, model, mu, sd in pairs:
                idx = np.where(mask)[0]
                if idx.size:
                    m_i, s_i = predict_nn(model, feats, idx, mu, sd, device)
                    p[idx] = m_i
                    s[idx] = s_i * std_factor   # t scale -> 1-sigma std (1.0 for gauss)
            return p, s

        # held-out (honest): each fold predicted by the OTHER fold; rest by all-train
        plx_sp, err_sp = predict_into([(A, model_B, muB, sdB),
                                       (B, model_A, muA, sdA),
                                       (rest, model_all, muAll, sdAll)])
        # in-fold (optimistic): each train fold by ITS OWN model -> overfit gauge
        plx_in, err_in = predict_into([(A, model_A, muA, sdA),
                                       (B, model_B, muB, sdB)])

        dist_kpc = 1.0 / plx_sp
        pd.DataFrame({
            "sdss_id": ids_k,
            "plx": plx_k, "e_plx": err_k,
            "plx_raw": S["plx_raw"], "zeropoint": S["zeropoint"],
            "plx_sp": plx_sp, "err_sp": err_sp,
            "dist_sp_kpc": dist_kpc,
            "r_med_photogeo_pc": S["dist_bj"],
            "sample": samp_k, "train": train_k,
            "spec_bad_frac": S["star_bad"],
        }).to_parquet(out_path, index=False)

        save_nn_checkpoint(model_out, model_all, muAll, sdAll, S["good"],
                           hidden=hidden, dropout=args.dropout,
                           dataset=spec_ds, snr_min=args.snr_min,
                           bad_frac=args.bad_frac, seed=args.seed,
                           std_factor=std_factor, **meta)
        log(f"wrote {out_path}  ({len(plx_k)} stars); saved model -> {model_out}")

        # ---- eval (same metrics as the linear baseline) + calibration + overfit gap ----
        def make_cat(p, e):
            return {"plx_a": plx_k[train_k], "err_a": err_k[train_k],
                    "plx_sp": p[train_k], "err_sp": e[train_k],
                    "train": np.ones(int(train_k.sum()), bool),
                    "sample": samp_k[train_k], "id": ids_k[train_k]}
        cat, cat_in = make_cat(plx_sp, err_sp), make_cat(plx_in, err_in)
        label = f"het-NN ({tag})"
        rep = E.evaluate(cat, label=label)
        rep_in = E.evaluate(cat_in, label="in-fold")
        gap = E.overfit_gap(rep_in, rep)
        bins, c = E.calibrate(cat)
        print()
        E.print_report(rep)
        E.print_overfit_gap(rep_in, rep)
        E.print_calibration(bins, c)

        # ---- per-star error head: is it real, calibrated, or just clamped? ----
        # (held-out cat; run sharpness FIRST -- rho~0 sinks the whole het motivation)
        ca = (cat["plx_sp"], cat["plx_a"], cat["err_a"], cat["err_sp"])
        rho = E.sharpness(*ca)
        z = E.zscore(*ca)
        z_width, z_tail = E.robust_scatter(z), float(np.mean(np.abs(z) > 3))
        cov = E.coverage(*ca)
        at_lo, at_hi = clamp_saturation(err_sp, std_factor)
        print(f"  sharpness (rho: err_sp vs |resid|, hi-S/N) : {rho:+.3f}"
              "   (~0 => het head decorative, a constant err would score as well)")
        print(f"  z-score honesty: robust width {z_width:.2f} (1=honest),"
              f" |z|>3 {100*z_tail:.2f}% (Gaussian 0.27% -> excess = tail)")
        E.print_coverage(cov)
        print(f"  sigma clamp saturation: {100*at_lo:.2f}% at floor (1e-4 mas),"
              f" {100*at_hi:.2f}% at ceil (10 mas)"
              "   (non-trivial => clamp doing model work)")
        print()
        for f in ("A", "B"):
            E.print_report(E.evaluate(cat, fold=f, label=label))
            E.print_calibration(*E.calibrate(cat, fold=f))
            print()

        if run is not None:
            summary = {
                "scatter": rep["robust_frac_scatter"],
                "bias": rep["median_frac_resid"],
                "chi2_mean": rep["chi2_mean_chi2"],
                "chi2_robust": rep["chi2_robust_chi2"],
                "median_spec_err_frac": rep["median_spec_err_frac"],
                "overfit_gap_pp": gap["gap_pp"],
                "recal_factor": c,
                "sharpness_rho": rho,
                "z_width": z_width,
                "z_tail_gt3": z_tail,
                "coverage_1sig": cov[0]["emp"],
                "clamp_floor_frac": at_lo,
                "clamp_ceil_frac": at_hi,
            }
            run.log(summary)
            run.summary.update(summary)
            run.finish()

    # ---- build the run list: a single config, or a gauss beta sweep ----
    if args.beta_sweep:
        betas = [float(b) for b in args.beta_sweep.split(",") if b.strip()]
        configs = [("gauss", b, args.nu) for b in betas]
        log(f"beta sweep over {betas} (gauss); sample prep reused across all configs")
    else:
        configs = [(args.loss, args.beta, args.nu)]

    base, ext = os.path.splitext(args.out)
    for kind, beta, nu in configs:
        loss_fn, std_factor, tag = make_loss(kind, beta, nu)
        single = len(configs) == 1
        out_path = args.out if single else f"{base}_{tag}{ext}"
        model_out = (args.model_out if (single and args.model_out)
                     else os.path.splitext(out_path)[0] + "_model.pt")
        run_config(loss_fn, std_factor, tag, out_path, model_out,
                   {"loss": kind, "beta": beta, "nu": nu})


if __name__ == "__main__":
    main()
