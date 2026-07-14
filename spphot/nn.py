"""
spphot.nn — heteroscedastic neural-net spectrophotometric parallax.

The model/loss/training/inference library behind train_nn.py and apply_nn.py
(same sample and A/B protocol as the linear model; see those drivers and
NN_MODEL.md for the full method description):

    x = standardized [photometry | ln-flux]  ->  HetMLP  ->  (z_mu, z_logs)
    plx_sp = exp(z_mu)   sig_sp = exp(z_logs)          (both positive)
    plx_gaia ~ Normal(plx_sp, sig_sp^2 + e_plx^2)      (het_nll; beta-NLL option)
    or Student-t with nu dof                           (studentt_nll, robust)

Also owns the torch-checkpoint schema: save_nn_checkpoint() writes it,
load_bundle() reads it back (with the telescope lit-masks) into a reusable
bundle, and apply_to_parquet() runs inference on new spectra with training-
identical feature assembly.
"""
from __future__ import annotations
import math, time
import numpy as np

from spphot.data import log, load_metadata, build_lnflux_streaming, load_pixel_masks

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
# checkpoint I/O + apply-time inference (this module owns the .pt schema)
# ----------------------------------------------------------------------
def save_nn_checkpoint(path, model, mu, sd, good, *, hidden, dropout, label_cols,
                       snr_min, bad_frac, seed, std_factor, **meta):
    """Write the train_nn checkpoint: weights + standardization + good-pixel mask
    + feature provenance, so load_bundle()/apply_to_parquet() reproduce the exact
    training preprocessing on new stars."""
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu, "sd": sd,
        "good_pixel_mask": good,
        "hidden": list(hidden), "dropout": dropout,
        "label_cols": list(label_cols), "d_in": int(mu.shape[0]),
        "snr_min": snr_min, "bad_frac": bad_frac, "seed": seed,
        "std_factor": std_factor, **meta,
    }, path)


def _resolve_cluster(parquet, allstar, col, sdss_ids):
    """Cluster-name array aligned to sdss_ids, reading `col` from the parquet
    scalar columns or (failing that) the allStar table."""
    import pyarrow.parquet as pq, pandas as pd
    from astropy.io import fits
    if col in set(pq.ParquetFile(parquet).schema_arrow.names):
        t = pq.read_table(parquet, columns=["sdss_id", col]).to_pandas()
    else:
        a = fits.open(allstar)[2].data
        def native(x):
            x = np.asarray(x)
            return x.astype(x.dtype.newbyteorder("=")) if x.dtype.byteorder == ">" else x
        t = pd.DataFrame({"sdss_id": native(a["sdss_id"]), col: native(a[col])}
                         ).drop_duplicates("sdss_id")
    lut = dict(zip(t["sdss_id"].to_numpy(), t[col].to_numpy()))
    return np.array([lut.get(s) for s in sdss_ids], dtype=object)


def load_bundle(model_path, pixel_mask_dir=None, device="auto"):
    """Load a train_nn checkpoint + the telescope lit-masks ONCE, into a reusable
    bundle. Build it once and pass it to apply_to_parquet() for every cluster so the
    (expensive) checkpoint load and HetMLP construction are not repeated per file."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # weights_only=False: our own checkpoint stores numpy arrays (good_pixel_mask,
    # mu/sd) that PyTorch 2.6's restricted unpickler rejects.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    good = np.asarray(ckpt["good_pixel_mask"], bool)
    mu = np.asarray(ckpt["mu"], np.float32)
    sd = np.asarray(ckpt["sd"], np.float32)
    std_factor = float(ckpt.get("std_factor", 1.0))
    label_cols = list(ckpt["label_cols"])
    hidden = tuple(ckpt["hidden"])
    dropout = float(ckpt.get("dropout", 0.0))
    d_in = int(ckpt["d_in"])
    log(f"loaded NN: good pixels={int(good.sum())}, d_in={d_in}, hidden={hidden}, "
        f"std_factor={std_factor:.3f}, loss={ckpt.get('loss')}, device={device}")

    model = HetMLP(d_in, hidden, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "model": model, "good": good, "mu": mu, "sd": sd,
        "std_factor": std_factor, "label_cols": label_cols, "d_in": d_in,
        "tel_masks": load_pixel_masks(pixel_mask_dir), "device": device,
    }


def apply_to_parquet(bundle, parquet, allstar, *, out=None, cluster_name=None,
                     cluster_col=None, f_max=2.0, batch_rows=20000):
    """Infer plx_sp/err_sp for one spectra parquet with a preloaded `bundle`.

    Mirrors training feature assembly exactly (saved pixel mask + telescope lit
    masks + saved standardization). Writes `out` if given and returns the result
    DataFrame so a caller can concatenate clusters for the test without re-reading.
    """
    import pandas as pd

    model, good, mu, sd = bundle["model"], bundle["good"], bundle["mu"], bundle["sd"]
    std_factor, label_cols, d_in = bundle["std_factor"], bundle["label_cols"], bundle["d_in"]
    tel_masks, device = bundle["tel_masks"], bundle["device"]

    merged, n_parquet = load_metadata(parquet, allstar)

    phot_all = merged[label_cols].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    log(f"keep (complete photometry): {int(keep.sum())} / {n_parquet}")

    # spectra on the SAVED pixel mask + telescope lit masks (identical to training)
    X_spec, _, star_bad = build_lnflux_streaming(
        parquet, keep, f_max=f_max, batch_rows=batch_rows,
        fixed_good=good, tel_masks=tel_masks)

    feats = np.hstack([phot_all[keep].astype(np.float32), X_spec]).astype(np.float32, copy=False)
    assert feats.shape[1] == d_in, (
        f"feature dim {feats.shape[1]} != model d_in {d_in} "
        f"(pixel-mask / photometry mismatch with training)")

    log("predicting with the all-train model ...")
    plx_sp, sig_sp = predict_nn(model, feats, np.arange(feats.shape[0]), mu, sd, device)
    err_sp = sig_sp * std_factor
    dist_kpc = 1.0 / plx_sp

    # Gaia targets, zero-point corrected like training (NaN where no zeropoint)
    plx_raw = merged["plx"].to_numpy(float)[keep]
    zpt = merged["zeropoint"].to_numpy(float)[keep]
    plx_corr = plx_raw - zpt
    n_zpt = int(np.isfinite(zpt).sum())
    if n_zpt == 0:
        log("no zeropoint in parquet -> plx = plx_raw; pass the global offset to "
            "the cluster test")
        plx_corr = plx_raw
    ids_k = merged["sdss_id"].to_numpy()[keep]

    cols = {
        "sdss_id": ids_k,
        "plx": plx_corr, "e_plx": merged["e_plx"].to_numpy(float)[keep],
        "plx_raw": plx_raw, "zeropoint": zpt,
        "plx_sp": plx_sp, "err_sp": err_sp, "dist_sp_kpc": dist_kpc,
        "r_med_photogeo_pc": merged["r_med_photogeo"].to_numpy(float)[keep],
        "spec_bad_frac": star_bad,
    }
    # carry membership probability through (added by prepare_cluster_spectra) so the
    # cluster test can threshold / probability-weight on it
    import pyarrow.parquet as pq
    if "memberprob" in set(pq.ParquetFile(parquet).schema_arrow.names):
        mp = pq.read_table(parquet, columns=["memberprob"]).column(0).to_numpy()
        cols["memberprob"] = np.asarray(mp, float)[keep]

    if cluster_col:
        cols["cluster"] = _resolve_cluster(parquet, allstar, cluster_col, ids_k)
    elif cluster_name:
        cols["cluster"] = np.full(int(keep.sum()), cluster_name)

    df = pd.DataFrame(cols)
    if out is not None:
        df.to_parquet(out, index=False)
        log(f"wrote {out}  ({int(keep.sum())} stars"
            + (f", {len(set(cols['cluster']))} clusters)" if "cluster" in cols else ")"))
    return df
