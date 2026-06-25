#!/usr/bin/env python3
"""
train_nn.py — heteroscedastic neural-net spectrophotometric parallax.

Same sample, features, zero-point, pixel masks and A/B split as run_full_gadi.py
(it reuses run_full_gadi.prepare_sample), but instead of the linear exp(X@theta)
fit it trains an MLP that outputs BOTH a parallax AND a per-star error. A learned,
per-star sigma is exactly what the linear baseline lacks: with a single constant
fractional error the eval shows mean chi2 ~ 2.6 (outlier-driven). A heteroscedastic
Gaussian head + scalar recal makes the ROBUST chi2 ~ 1 and usually tightens the
scatter, but leaves mean chi2 >> 1 -- that outlier tail is only shrunk by the
Student-t head (studentt_nll). Credit each claim to the right mechanism.

Model (features x = standardized [photometry | ln-flux on the good pixels]):
    body  = MLP(x)              -> (z_mu, z_logs)
    plx_sp = exp(z_mu)          positive spec parallax (log-space, like the linear model)
    sig_sp = exp(z_logs)        positive per-star spec-parallax error
Likelihood (parallax space, never inverted; negative Gaia parallaxes kept):
    plx_gaia ~ Normal(plx_sp, sig_sp^2 + e_plx^2)
    NLL = 0.5 * [ (plx - plx_sp)^2 / (sig_sp^2 + e_plx^2) + log(sig_sp^2 + e_plx^2) ]
The Gaia measurement error e_plx is folded into the variance, so sig_sp learns the
*intrinsic* spec-parallax scatter.

Cross-validation mirrors run_full_gadi exactly: a fold-A model, a fold-B model, and
an all-train model; each training star is predicted by the model that did NOT see its
fold, all others by the all-train model. The results parquet uses the same schema, so
spphot_eval scores it apples-to-apples against the linear baseline.

The eval report now also prints an OVERFIT GAP (robust scatter from in-fold vs
held-out predictions): the MLP can overfit where the linear model can't, so a
large gap says trust only the held-out number and regularize harder.

Usage (see train_nn.pbs):
  python train_nn.py --parquet <spectra_zpt.parquet> --allstar <allStar.fits> \
        --pixel-mask-dir <dir> --out nn_results.parquet \
        [--hidden 1024,512,256] [--epochs 60] [--batch-size 4096] [--device auto]

  # beta-NLL calibration<->bias sweep (one sample prep, one model per beta):
  python train_nn.py ... --out nn.parquet --beta-sweep 0.2,0.3,0.5,0.7
  #   -> nn_gauss-beta0.2.parquet, ... ; compare bias vs robust chi2 across them
"""
from __future__ import annotations
import os, sys, math, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_full_gadi as R   # prepare_sample, log, LABEL_COLS
from run_full_gadi import log

import torch
import torch.nn as nn
import torch.nn.functional as F

# sig_sp is clamped to this range (mas) for numerical safety in exp/log
LOG_SIG_MIN, LOG_SIG_MAX = math.log(1e-4), math.log(10.0)


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
class HetMLP(nn.Module):
    """MLP with two linear heads: log-parallax mean and log spec-error."""

    def __init__(self, d_in, hidden=(1024, 512, 256), dropout=0.1):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head_mu = nn.Linear(d, 1)      # z_mu   -> plx_sp = exp(z_mu)
        self.head_logs = nn.Linear(d, 1)    # z_logs -> sig_sp = exp(z_logs)

    def forward(self, x):
        z = self.body(x)
        return self.head_mu(z).squeeze(-1), self.head_logs(z).squeeze(-1)


def _mu_sig(z_mu, z_logs):
    """Map raw heads to (plx_sp, sig_sp) with the safety clamps."""
    m = torch.exp(torch.clamp(z_mu, -30.0, 30.0))
    sig = torch.exp(torch.clamp(z_logs, LOG_SIG_MIN, LOG_SIG_MAX))
    return m, sig


def clamp_saturation(err_sp, std_factor, tol=0.01):
    """Fraction of predicted sigmas pinned at the LOG_SIG_MIN/MAX rails.

    err_sp is the 1-sigma std (sig * std_factor); divide it back to recover the
    clamped sig and test proximity to the [exp(LOG_SIG_MIN), exp(LOG_SIG_MAX)]
    bounds. A non-trivial fraction at a rail means the clamp -- not the data -- is
    setting those errors, so sharpness/calibration there are partly measuring the
    clamp; widen the bound or check why the head wants to run off it."""
    sig = np.asarray(err_sp, float) / std_factor
    sig = sig[np.isfinite(sig)]
    lo, hi = math.exp(LOG_SIG_MIN), math.exp(LOG_SIG_MAX)
    if sig.size == 0:
        return 0.0, 0.0
    return (float(np.mean(sig <= lo * (1 + tol))),
            float(np.mean(sig >= hi * (1 - tol))))


def het_nll(z_mu, z_logs, plx, err, beta=0.0):
    """Gaussian-in-parallax NLL with the Gaia error folded into the variance.

    beta>0 gives the beta-NLL of Seitzer et al. 2022: each star's loss is weighted by
    detach(var)**beta. Plain NLL (beta=0) down-weights high-variance stars and so
    UNDERFITS the mean there (-> biased, unstable variance head); beta=0.5 restores
    the mean's gradient in those regions (beta=1 ~ MSE) and removes most of that bias
    while keeping the calibration."""
    m, sig = _mu_sig(z_mu, z_logs)
    var = sig * sig + err * err
    nll = 0.5 * ((plx - m) ** 2 / var + torch.log(var))
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()


def studentt_nll(z_mu, z_logs, plx, err, nu):
    """Heteroscedastic Student-t NLL: same per-star (plx_sp, sig_sp) heads, but the
    residual is modelled as Student-t with squared scale sig_sp^2 + e_plx^2 and nu
    degrees of freedom. The fat tails downweight outliers during training (a binary
    contributes ~log of its residual instead of its square), so the fit and the
    per-star scale are not dragged by the tail. sig_sp here is the t SCALE; its
    standard deviation is sig_sp * sqrt(nu/(nu-2)) (applied at output time)."""
    m, sig = _mu_sig(z_mu, z_logs)
    s2 = sig * sig + err * err
    r2 = (plx - m) ** 2 / s2
    c = math.lgamma(nu / 2.0) - math.lgamma((nu + 1.0) / 2.0) + 0.5 * math.log(nu * math.pi)
    return (c + 0.5 * torch.log(s2) + 0.5 * (nu + 1.0) * torch.log1p(r2 / nu)).mean()


# ----------------------------------------------------------------------
# standardize [phot | spec] with train-fold statistics
# ----------------------------------------------------------------------
def standardize_fit(A):
    mu = A.mean(0)
    sd = A.std(0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# ----------------------------------------------------------------------
# train / predict
# ----------------------------------------------------------------------
def train_fold(feats, train_mask, plx, err, mu, sd, *, loss_fn, hidden, dropout, lr,
               weight_decay, epochs, batch_size, device, seed, label, log_cb=None):
    """Train a HetMLP on the stars in train_mask. Features are standardized per batch
    with the supplied (mu, sd) — the train-fold stats — so nothing leaks from the
    held-out fold."""
    torch.manual_seed(seed)
    idx_all = np.where(train_mask)[0]
    n = idx_all.size
    d_in = feats.shape[1]

    model = HetMLP(d_in, hidden, dropout).to(device)
    with torch.no_grad():                                  # sensible head biases
        ptr = plx[idx_all]
        med = float(np.median(ptr[ptr > 0])) if np.any(ptr > 0) else 1.0
        model.head_mu.bias.fill_(math.log(max(med, 1e-3)))
        model.head_logs.bias.fill_(math.log(0.1))          # ~0.1 mas initial spec error

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)
    feats_t = torch.from_numpy(feats)                      # CPU; batches move to device
    plx_t = torch.from_numpy(plx.astype(np.float32))
    err_t = torch.from_numpy(err.astype(np.float32))
    idx_t = torch.from_numpy(idx_all)

    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = idx_t[torch.randperm(n, generator=g)]
        running = 0.0
        for i in range(0, n, batch_size):
            bidx = perm[i:i + batch_size]
            xb = ((feats_t[bidx].to(device) - mu_t) / sd_t)
            yb = plx_t[bidx].to(device)
            eb = err_t[bidx].to(device)
            z_mu, z_logs = model(xb)
            loss = loss_fn(z_mu, z_logs, yb, eb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * bidx.numel()
        sched.step()
        if log_cb is not None:
            log_cb(label, ep + 1, running / n, sched.get_last_lr()[0])
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            log(f"    [{label}] epoch {ep+1}/{epochs}  NLL={running/n:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)")
    return model


def predict_nn(model, feats, idx, mu, sd, device, batch=65536):
    """Return (plx_sp, sig_sp) for feats[idx], standardized with (mu, sd)."""
    model.eval()
    mu_t = torch.from_numpy(mu).to(device)
    sd_t = torch.from_numpy(sd).to(device)
    feats_t = torch.from_numpy(feats)
    idx = np.asarray(idx)
    out_m = np.empty(idx.size, np.float64)
    out_s = np.empty(idx.size, np.float64)
    with torch.no_grad():
        for i in range(0, idx.size, batch):
            bidx = torch.from_numpy(idx[i:i + batch])
            xb = (feats_t[bidx].to(device) - mu_t) / sd_t
            z_mu, z_logs = model(xb)
            m, sig = _mu_sig(z_mu, z_logs)
            out_m[i:i + batch] = m.cpu().numpy()
            out_s[i:i + batch] = sig.cpu().numpy()
    return out_m, out_s


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
    import spphot_eval as E

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1)))
    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    log(f"device={device}  hidden={hidden}  epochs={args.epochs}  batch={args.batch_size}")

    # ---- identical sample/features/splits as the linear pipeline. Built ONCE and
    #      reused across every sweep config (this is the expensive step). ----
    S = R.prepare_sample(args.parquet, args.allstar, snr_min=args.snr_min,
                         bad_frac=args.bad_frac, batch_rows=args.batch_rows,
                         pixel_mask_dir=args.pixel_mask_dir, seed=args.seed)
    spec = S["spec"]
    # features = [photometry | ln-flux]; float32 for the net (frees the separate spec)
    feats = np.hstack([S["phot"].astype(np.float32), spec]).astype(np.float32, copy=False)
    del spec, S["spec"]
    plx_k, err_k = S["plx"], S["err"]
    samp_k, train_k = S["sample"], S["train"]
    ids_k = S["ids"]
    log(f"features: {feats.shape[0]} stars x {feats.shape[1]} (phot 8 + spec {feats.shape[1]-8})")

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

        torch.save({
            "state_dict": model_all.state_dict(),
            "mu": muAll, "sd": sdAll,
            "good_pixel_mask": S["good"],
            "hidden": list(hidden), "dropout": args.dropout,
            "label_cols": list(R.LABEL_COLS), "d_in": int(feats.shape[1]),
            "snr_min": args.snr_min, "bad_frac": args.bad_frac, "seed": args.seed,
            "std_factor": std_factor, **meta,
        }, model_out)
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
