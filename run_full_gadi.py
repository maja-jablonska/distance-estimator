#!/usr/bin/env python3
"""
run_full_gadi.py — full-dataset spectrophotometric-parallax run (Hogg+18 style)
for the ~800k-star APOGEE bulge sample, in a single Gadi batch job.

Pipeline (all in one process):
  1. load allStar FITS metadata (plx, e_plx, 8 magnitudes, Bailer-Jones distance)
  2. merge onto the spectra parquet by sdss_id (left join -> preserves parquet row order)
  3. build the continuum-normalized ln-flux matrix in ONE streaming pass over the
     parquet (chunked, float32). Shared good-pixel mask from the ivar==0 masked
     sentinel + flux/continuum sanity. Peak RAM ~ N_keep * 8575 * 4 bytes.
  4. quality-cut training set + reproducible A/B split (seeded)
  5. fit THREE models — fold-A, fold-B, all-training — each a log-space ridge
     warm-start + parallax-space L-BFGS refine (jitted value+grad)
  6. predict spec parallax for EVERY kept star:
       training stars -> cross-validated (A predicted by B, B by A)  [honest]
       all other stars -> the all-training model
  7. constant fractional err_sp from the hi-S/N probe out-of-sample scatter
  8. write a results parquet (+ npz of fitted theta/stats) and print the eval report

Why this differs from the notebook: build_sample_from_parquet vstacks flux+continuum
+ivar for every row at once (~165 GB at 800k). Here we stream, keep only ln-flux as
float32, and never hold the full flux/continuum/ivar arrays. Request a >=96 GB node.

Usage (see run_full_gadi.pbs):
  python run_full_gadi.py --parquet <spectra.parquet> --allstar <allStar.fits> \
                          --out results_full.parquet [--lam 0.1] [--batch-rows 20000]
"""
from __future__ import annotations
import os, sys, time, argparse

# The fit is small dense ridge + L-BFGS — run it on CPU. Forcing this before JAX is
# imported avoids the GPU/CuDNN probe that crashes on a CuDNN runtime/compile version
# mismatch (RET_CHECK failure ... dnn_support != nullptr). setdefault so an explicit
# JAX_PLATFORMS in the environment still wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # for spphot_eval / spphot_plots

LABEL_COLS = ["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
META_COLS  = ["sdss_id", "plx", "e_plx", *LABEL_COLS, "r_med_photogeo"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# per-telescope "lit-anywhere" pixel masks (build_pixel_mask.py products)
# ----------------------------------------------------------------------
# A pixel is "lit" (True) when at least one spectrum from that telescope flagged
# it; the never-lit pixels are chip gaps / no-coverage. We KEEP only lit pixels
# (good_telescope = lit_mask) and intersect that, per star, with the existing
# per-pixel data-quality bad detection (ivar==0 / continuum sanity). Telescopes
# carry their own grid, so masks are stored one .npy per telescope.
TELESCOPES = ("apo1m", "apo25m", "lco25m")


def load_pixel_masks(mask_dir, telescopes=TELESCOPES):
    """{telescope: bool lit-mask} from <mask_dir>/pixel_lit_mask_<tel>.npy.

    Missing files are skipped (those telescopes get no extra masking). Returns
    None if mask_dir is falsy, so the caller stays backward-compatible."""
    if not mask_dir:
        return None
    masks = {}
    for t in telescopes:
        p = os.path.join(mask_dir, f"pixel_lit_mask_{t}.npy")
        if os.path.exists(p):
            m = np.load(p).astype(bool)
            masks[t] = m
            log(f"pixel mask {t}: keep {int(m.sum())}/{m.size} lit pixels ({p})")
        else:
            log(f"pixel mask {t}: MISSING ({p}) -> no telescope masking for {t}")
    return masks or None


def load_apogee_windows(path, width):
    """HOOK (not yet wired in): build a (width,) bool mask that is True on the union
    of the APOGEE element windows, to restrict the fit to spectral-line regions.

    To enable: load the per-element window definitions (e.g. the apogee/aspcap line
    list or the global_mask / element-window FITS) into a boolean array of length
    `width`, then AND it into each telescope's lit mask in load_pixel_masks (so
    `good_telescope = lit_mask & window_mask`). Left unimplemented on purpose — wire
    it up once the window file path/format is settled."""
    raise NotImplementedError("APOGEE element-window masking not wired up yet")


# ----------------------------------------------------------------------
# metadata: allStar FITS  +  parquet scalar columns, merged on sdss_id
# ----------------------------------------------------------------------
def load_metadata(parquet_path, allstar_path):
    import pandas as pd
    import pyarrow.parquet as pq
    from astropy.io import fits

    log("reading allStar metadata table ...")
    a = fits.open(allstar_path)[2].data            # extension 2 holds the table
    allstar = pd.DataFrame({c: np.asarray(a[c]) for c in META_COLS})
    for c in allstar.columns:                      # FITS is big-endian; pandas wants native
        arr = allstar[c].values
        if getattr(arr.dtype, "byteorder", "=") == ">":
            allstar[c] = arr.astype(arr.dtype.newbyteorder("="))

    # astra allStar has one row per spectrum/reduction, so a star (sdss_id) can
    # appear several times; META_COLS are per-star quantities, so keep one row
    # per star — preferring the most complete one in case duplicates carry NaNs.
    n_dup = int(allstar["sdss_id"].duplicated().sum())
    if n_dup:
        completeness = allstar[["plx", "e_plx", *LABEL_COLS]].notna().sum(axis=1)
        allstar = (allstar.assign(_complete=completeness)
                   .sort_values("_complete", ascending=False, kind="stable")
                   .drop_duplicates("sdss_id")
                   .drop(columns="_complete"))
        log(f"allStar: collapsed {n_dup} duplicate sdss_id rows "
            f"-> {len(allstar)} unique stars")

    log("reading parquet scalar columns (sdss_id, snr, spectrum_flags, zeropoint) ...")
    avail = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    scalar_cols = ["sdss_id", "snr", "spectrum_flags"]
    has_zpt = "zeropoint" in avail
    if has_zpt:
        scalar_cols.append("zeropoint")
    meta = pq.read_table(parquet_path, columns=scalar_cols).to_pandas()
    if not has_zpt:
        meta["zeropoint"] = np.nan
        log("parquet has no 'zeropoint' column -> parallax zero-point correction disabled")
    n_parquet = len(meta)

    # left join keeps parquet row order, so it stays aligned to the streamed spectra
    merged = meta.merge(allstar, on="sdss_id", how="left")
    assert len(merged) == n_parquet, "merge changed row count (duplicate sdss_id in allStar?)"
    log(f"metadata: {n_parquet} parquet rows, "
        f"{merged['plx'].notna().sum()} with allStar match")
    return merged, n_parquet


# ----------------------------------------------------------------------
# streaming spectral builder (the memory-critical part)
# ----------------------------------------------------------------------
def _list_col_2d(arr, width):
    """ListArray of fixed-length float sublists -> (n, width) float64, fast path."""
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    assert flat.size % width == 0, "ragged spectral column (non-fixed length)"
    return flat.reshape(-1, width)


def build_lnflux_streaming(parquet_path, keep_mask, f_max=2.0, bad_frac_max=0.01,
                           batch_rows=20000, fixed_good=None, tel_masks=None):
    """One streaming pass. Returns:
        X_spec    (n_keep, L) float32  ln(normalized flux) on shared good pixels
        good      (Lfull,)    bool     kept-pixel mask
        star_bad  (n_keep,)   float32  per-star bad-pixel fraction (data-quality flag)
    keep_mask is in PARQUET ROW ORDER (same order as load_metadata's merged frame).
    If fixed_good is given (a saved model's mask), it is used verbatim instead of
    recomputing — so new spectra land on exactly the pixels the model was fit on.
    If tel_masks (a {telescope: lit-mask} dict from load_pixel_masks) is given, a
    pixel not lit by a star's telescope is treated as bad for that star (intersected
    with the data-quality bad detection), so the feature only ever uses lit pixels."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n_keep = int(keep_mask.sum())
    lnfull = None
    bad_count = None
    star_bad = np.empty(n_keep, np.float32)
    width = None
    out = 0          # next free row in the kept-output arrays
    row0 = 0         # running offset into the full parquet
    t0 = time.time()
    unknown_tels = set()

    columns = ["flux", "continuum", "ivar"]
    if tel_masks is not None:
        columns.append("telescope")

    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows,
                                               columns=columns)):
        bsz = batch.num_rows
        sel = keep_mask[row0:row0 + bsz]
        row0 += bsz
        k = int(sel.sum())
        if k == 0:
            continue
        if width is None:
            width = batch.column("flux").values.to_numpy(zero_copy_only=False).size // bsz
            lnfull = np.zeros((n_keep, width), np.float32)
            bad_count = np.zeros(width, np.int64)
            if tel_masks is not None:
                for t, m in tel_masks.items():
                    if m.size != width:
                        raise ValueError(f"pixel mask {t} length {m.size} != grid {width}")

        flux = _list_col_2d(batch.column("flux"), width)[sel]
        cont = _list_col_2d(batch.column("continuum"), width)[sel]
        ivar = _list_col_2d(batch.column("ivar"), width)[sel]

        C = np.where(cont > 0, cont, np.nan)
        f = flux / C                                              # normalized flux
        bad = (~np.isfinite(f) | ~np.isfinite(ivar) | (ivar <= 0)
               | (cont <= 0) | (f <= 0) | (f > f_max))            # ivar==0 is APOGEE's mask sentinel
        star_bad[out:out + k] = bad.mean(axis=1)                  # quality flag, telescope-independent

        if tel_masks is not None:                                 # keep only telescope-lit pixels (AND)
            tels = np.asarray(batch.column("telescope").to_pylist())[sel]
            lit = np.ones((k, width), bool)
            for t in np.unique(tels):
                m = tel_masks.get(str(t))
                if m is None:
                    unknown_tels.add(str(t))
                    continue                                      # unknown telescope -> no extra masking
                lit[tels == t] = m
            bad |= ~lit

        f = np.where(bad, 1.0, f)                                 # impute bad -> continuum (ln 0)
        lnfull[out:out + k] = np.log(f).astype(np.float32)
        bad_count += bad.sum(axis=0)
        out += k

        if bi % 20 == 0:
            log(f"  spectra: {out}/{n_keep} kept rows "
                f"({(time.time()-t0):.0f}s)")

    if unknown_tels:
        log(f"WARNING: no pixel mask for telescopes {sorted(unknown_tels)} "
            f"-> those spectra kept all data-quality-good pixels")

    assert out == n_keep, f"filled {out} rows, expected {n_keep}"
    if fixed_good is not None:
        assert fixed_good.size == width, "saved mask length != spectral grid"
        good = fixed_good
        log(f"spectra done: grid={width}, applying saved mask L={int(good.sum())}")
    else:
        good = bad_count < bad_frac_max * n_keep
        log(f"spectra done: grid={width}, good pixels kept={int(good.sum())} "
            f"(Hogg+18 kept 7405)")
    X_spec = lnfull[:, good].copy()                              # (n_keep, L) float32
    del lnfull                                                   # free the full grid
    return X_spec, good, star_bad


# ----------------------------------------------------------------------
# model: standardize + design, robust fit, batched predict
# ----------------------------------------------------------------------
def design(phot, spec, stats=None):
    """Standardize [phot | spec] with given (train) stats, prepend the intercept."""
    A = np.hstack([np.asarray(phot, float), np.asarray(spec, float)])
    if stats is None:
        mu, sd = A.mean(0), A.std(0)
        sd[sd < 1e-8] = 1.0
        stats = (mu, sd)
    mu, sd = stats
    A = (A - mu) / sd
    return np.hstack([np.ones((len(A), 1)), A]), stats


def fit_parallax_model(X, plx, sigma, lam, maxiter=2000):
    """theta for plx ~ exp(X @ theta): Gaussian-in-parallax NLL + L2 ridge.
    Log-space ridge warm-start (positive parallaxes), then parallax-space refine."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import minimize

    X = np.asarray(X, float); plx = np.asarray(plx, float); sigma = np.asarray(sigma, float)
    N, D = X.shape
    reg = np.full(D, lam); reg[0] = 0.0

    pos = plx > 0
    Xw, yw = X[pos], np.log(plx[pos])
    try:
        theta0 = np.linalg.solve(Xw.T @ Xw + np.diag(reg) * pos.sum(), Xw.T @ yw)
    except np.linalg.LinAlgError:
        theta0 = np.zeros(D); theta0[0] = np.log(np.median(plx[pos]))

    Xj, yj, sj, regj = (jnp.asarray(np.asarray(v, float)) for v in (X, plx, sigma, reg))
    invN = 1.0 / N

    # pass the data as traced args (not closed-over) so XLA does NOT bake the full
    # design matrix into the compiled fn as a multi-GB constant; it stays a device
    # buffer compiled once and reused every L-BFGS step.
    @jax.jit
    def value_and_grad(theta, Xj, yj, sj, regj):
        def obj(th):
            m = jnp.exp(jnp.clip(Xj @ th, -30, 30))
            r = (yj - m) / sj
            return 0.5 * invN * jnp.sum(r ** 2) + 0.5 * jnp.sum(regj * th ** 2)
        return jax.value_and_grad(obj)(theta)

    def scipy_obj(theta):
        fv, g = value_and_grad(jnp.asarray(theta), Xj, yj, sj, regj)
        return float(fv), np.asarray(g, np.float64)

    # The objective is mean-scaled (1/N), so its gradient is ~1/N smaller than the
    # raw sum. L-BFGS-B's default pgtol=1e-5 is then met after a single step on big
    # N (it "converges" at iters=1 without actually refining). Tighten gtol/ftol so
    # the parallax-space refine genuinely runs to a stationary point.
    res = minimize(scipy_obj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                            "gtol": 1e-9, "ftol": 1e-12})
    gnorm = float(np.max(np.abs(res.jac))) if res.jac is not None else float("nan")
    if res.nit <= 1:
        log(f"  WARNING: fit stopped at iters={res.nit} (|grad|inf={gnorm:.2e}); "
            f"refine may not have run — check tolerances")
    return res.x, res


def predict(theta, stats, phot, spec, batch=50000):
    """exp(theta . x) in batches so we never standardize the whole 800k at once."""
    out = np.empty(len(phot))
    for i in range(0, len(phot), batch):
        Xb, _ = design(phot[i:i + batch], spec[i:i + batch], stats)
        out[i:i + batch] = np.exp(np.clip(Xb @ theta, -30, 30))
    return out


def cv_fold_scatter(phot_tr, spec_tr, plx_tr, err_tr, fold_tr, lam, maxiter=2000):
    """Fit fold-A and fold-B at this lam, predict each fold with the OTHER fold's
    model, and return the honest cross-validated headline metric:
        (robust fractional scatter on the hi-S/N probe, (thA,stA), (thB,stB)).
    This is exactly the number print_report calls SCATTER, so it is the right
    quantity to choose lam on. The fold models are returned so the caller can
    reuse the winning lam's fits without refitting them."""
    import spphot_eval as E
    A, B = fold_tr == "A", fold_tr == "B"
    XA, stA = design(phot_tr[A], spec_tr[A])
    thA, _ = fit_parallax_model(XA, plx_tr[A], err_tr[A], lam=lam, maxiter=maxiter)
    XB, stB = design(phot_tr[B], spec_tr[B])
    thB, _ = fit_parallax_model(XB, plx_tr[B], err_tr[B], lam=lam, maxiter=maxiter)
    plx_sp = np.empty(len(plx_tr))
    plx_sp[A] = predict(thB, stB, phot_tr[A], spec_tr[A])   # A held out from B's fit
    plx_sp[B] = predict(thA, stA, phot_tr[B], spec_tr[B])   # B held out from A's fit
    probe = np.isfinite(plx_tr) & (plx_tr > 0) & (plx_tr / err_tr >= 20.0)
    scatter = E.robust_scatter(E.fractional_residuals(plx_sp[probe], plx_tr[probe]))
    return scatter, (thA, stA), (thB, stB)


# ----------------------------------------------------------------------
# model persistence — one self-contained, re-loadable artifact
# ----------------------------------------------------------------------
def save_model(path, *, theta_all, stats_all, theta_A, stats_A, theta_B, stats_B,
               good, frac_sigma, config, tel_masks=None):
    """Everything needed to predict on NEW stars without refitting: the all-train
    model (theta + standardization), the good-pixel mask, the adopted fractional
    error, the feature order, and the build config. The fold models are kept too
    so the A/B cross-validation is reproducible. The per-telescope pixel masks are
    embedded so apply-time imputation matches training. Load with load_model()."""
    extra = {}
    if tel_masks:                                  # one array per telescope + the tag list
        extra["tel_mask_tags"] = np.array(list(tel_masks))
        for t, m in tel_masks.items():
            extra[f"tel_mask_{t}"] = m
    np.savez_compressed(
        path,
        # primary (use this to predict new data)
        theta_all=theta_all, mu_all=stats_all[0], sd_all=stats_all[1],
        good_pixel_mask=good, frac_sigma=np.float64(frac_sigma),
        # fold models (for reproducing the cross-validated predictions)
        theta_A=theta_A, mu_A=stats_A[0], sd_A=stats_A[1],
        theta_B=theta_B, mu_B=stats_B[0], sd_B=stats_B[1],
        # provenance / build settings so apply-time normalization matches
        label_cols=np.array(LABEL_COLS),
        lam=np.float64(config["lam"]), f_max=np.float64(config["f_max"]),
        bad_frac=np.float64(config["bad_frac"]), snr_min=np.float64(config["snr_min"]),
        seed=np.int64(config["seed"]),
        **extra,
    )


def load_model(path):
    """Return a dict with theta_all, stats_all=(mu,sd), good_pixel_mask, frac_sigma,
    f_max, bad_frac, label_cols — the pieces apply_model.py needs."""
    z = np.load(path, allow_pickle=False)
    tel_masks = None
    if "tel_mask_tags" in z.files:
        tel_masks = {str(t): z[f"tel_mask_{t}"].astype(bool) for t in z["tel_mask_tags"]}
    return {
        "theta_all": z["theta_all"], "stats_all": (z["mu_all"], z["sd_all"]),
        "theta_A": z["theta_A"], "stats_A": (z["mu_A"], z["sd_A"]),
        "theta_B": z["theta_B"], "stats_B": (z["mu_B"], z["sd_B"]),
        "good_pixel_mask": z["good_pixel_mask"], "frac_sigma": float(z["frac_sigma"]),
        "f_max": float(z["f_max"]), "bad_frac": float(z["bad_frac"]),
        "label_cols": [str(c) for c in z["label_cols"]],
        "tel_masks": tel_masks,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--out", default="results_full.parquet")
    ap.add_argument("--model-out", default=None,
                    help="path for the saved model npz (default: <out>_model.npz)")
    ap.add_argument("--lam", type=float, default=0.1, help="ridge strength")
    ap.add_argument("--lam-scan", default=None,
                    help="comma-separated ridge values to scan, e.g. '0.003,0.01,0.03,0.1,0.3'; "
                         "picks the lam with the lowest cross-validated fold scatter, then "
                         "runs the full pipeline with it")
    ap.add_argument("--snr-min", type=float, default=100.0, help="training S/N cut")
    ap.add_argument("--bad-frac", type=float, default=0.01, help="shared good-pixel threshold")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--pixel-mask-dir", default=None,
                    help="dir with per-telescope pixel_lit_mask_<tel>.npy; keeps only "
                         "lit pixels (intersected with the data-quality mask)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import pandas as pd

    tel_masks = load_pixel_masks(args.pixel_mask_dir)
    merged, n_parquet = load_metadata(args.parquet, args.allstar)

    # ---- keep = complete photometry (we build spectra + predict for all of these) ----
    phot_all = merged[LABEL_COLS].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    log(f"keep (complete photometry): {keep.sum()} / {n_parquet}")

    # ---- training set: quality cuts only; NO cut on parallax sign or S/N ----
    # apply the Gaia DR3 parallax zero-point: plx_corr = plx - zeropoint. The
    # correction is undefined (NaN) for non-5/6-parameter sources -> plx_corr is
    # NaN there, so the np.isfinite(plx_corr) cut drops them from training.
    plx_raw = merged["plx"].to_numpy(float)
    zpt_all = merged["zeropoint"].to_numpy(float)
    plx_all = plx_raw - zpt_all
    err_all = merged["e_plx"].to_numpy(float)
    snr_ok  = merged["snr"].to_numpy(float)
    flags   = merged["spectrum_flags"].to_numpy()
    n_zpt = int(np.isfinite(zpt_all).sum())
    n_no_zpt = int((np.isfinite(plx_raw) & ~np.isfinite(zpt_all)).sum())
    log(f"zero-point: applied to {n_zpt} sources; "
        f"{n_no_zpt} stars have plx but no zeropoint (dropped from training)")
    train = (keep & (snr_ok > args.snr_min) & (flags == 0)
             & np.isfinite(plx_all) & np.isfinite(err_all) & (err_all > 0))

    # ---- reproducible 50/50 A/B split, stratified on train / non-train ----
    rng = np.random.default_rng(args.seed)
    sample = np.full(n_parquet, "B")
    for mask in (train, keep & ~train):
        idx = np.where(mask)[0]
        sample[idx[rng.permutation(len(idx))[:len(idx) // 2]]] = "A"
    log(f"training stars: {train.sum()}  | negative plx kept: "
        f"{100*(plx_all[train] < 0).mean():.1f}%")

    # ---- build spectra for every kept star (streaming) ----
    X_spec, good, star_bad = build_lnflux_streaming(
        args.parquet, keep, bad_frac_max=args.bad_frac, batch_rows=args.batch_rows,
        tel_masks=tel_masks)

    # ---- restrict metadata to kept rows (same order as X_spec) ----
    phot_k = phot_all[keep]
    plx_k, err_k = plx_all[keep], err_all[keep]      # plx_k is zero-point corrected
    plx_raw_k, zpt_k = plx_raw[keep], zpt_all[keep]
    samp_k = sample[keep]
    train_k = train[keep]
    ids_k = merged["sdss_id"].to_numpy()[keep]
    dist_bj_k = merged["r_med_photogeo"].to_numpy(float)[keep]

    # ---- fit three models on the training subset ----
    phot_tr, spec_tr = phot_k[train_k], X_spec[train_k]
    plx_tr, err_tr, fold_tr = plx_k[train_k], err_k[train_k], samp_k[train_k]

    def fit_on(mask, name, lam):
        Xf, st = design(phot_tr[mask], spec_tr[mask])
        th, res = fit_parallax_model(Xf, plx_tr[mask], err_tr[mask], lam=lam)
        log(f"  fit {name}: {mask.sum()} stars | lam={lam:g} converged={res.success} "
            f"iters={res.nit} obj={res.fun:.4f}")
        return th, st

    # ---- optional ridge scan: choose lam by cross-validated fold scatter ----
    fold_models = None
    if args.lam_scan:
        import re
        lams = [float(x) for x in re.split(r"[,;\s]+", args.lam_scan.strip()) if x]
        log(f"lambda scan over {lams} "
            f"(cross-validated robust fractional scatter on the hi-S/N probe) ...")
        scan = {}
        for lam in lams:
            sc, fa, fb = cv_fold_scatter(phot_tr, spec_tr, plx_tr, err_tr, fold_tr, lam)
            scan[lam] = (sc, fa, fb)
            log(f"  lam={lam:<8g} CV fold scatter = {100*sc:.2f}%")
        best_lam = min(scan, key=lambda L: scan[L][0])
        log("  --- lambda scan summary (lower is better) ---")
        for lam in lams:
            mark = "   <-- selected" if lam == best_lam else ""
            log(f"    lam={lam:<8g} {100*scan[lam][0]:.2f}%{mark}")
        args.lam = best_lam
        _, (theta_A, stats_A), (theta_B, stats_B) = scan[best_lam]   # reuse winning fits
        fold_models = (theta_A, stats_A, theta_B, stats_B)

    log("fitting models ...")
    if fold_models is not None:
        theta_A, stats_A, theta_B, stats_B = fold_models      # already fit at best lam
        log(f"  reusing fold-A/B fits from the lambda scan (lam={args.lam:g})")
    else:
        theta_A, stats_A = fit_on(fold_tr == "A", "fold-A", args.lam)
        theta_B, stats_B = fit_on(fold_tr == "B", "fold-B", args.lam)
    theta_all, stats_all = fit_on(np.ones(len(plx_tr), bool), "all-train", args.lam)

    # ---- predict every kept star ----
    log("predicting spec parallaxes ...")
    plx_sp = np.empty(keep.sum())
    A_tr = train_k & (samp_k == "A")          # training, fold A -> predicted by B
    B_tr = train_k & (samp_k == "B")          # training, fold B -> predicted by A
    rest = ~train_k                           # non-training -> all-train model
    plx_sp[A_tr] = predict(theta_B, stats_B, phot_k[A_tr], X_spec[A_tr])
    plx_sp[B_tr] = predict(theta_A, stats_A, phot_k[B_tr], X_spec[B_tr])
    plx_sp[rest] = predict(theta_all, stats_all, phot_k[rest], X_spec[rest])

    # ---- constant fractional err_sp from hi-S/N probe out-of-sample scatter ----
    import spphot_eval as E
    probe = (train_k & np.isfinite(plx_k) & (plx_k > 0)
             & (plx_k / err_k >= 20.0))
    frac_sigma = E.robust_scatter(np.log(plx_sp[probe]) - np.log(plx_k[probe]))
    err_sp = frac_sigma * plx_sp
    dist_kpc = 1.0 / plx_sp
    log(f"adopted fractional spec error: {100*frac_sigma:.1f}% "
        f"(probe N={probe.sum()})")

    # ---- write results ----
    out = pd.DataFrame({
        "sdss_id": ids_k,
        "plx": plx_k, "e_plx": err_k,            # plx is zero-point corrected (plx_raw - zeropoint)
        "plx_raw": plx_raw_k, "zeropoint": zpt_k,
        "plx_sp": plx_sp, "err_sp": err_sp,
        "dist_sp_kpc": dist_kpc,
        "r_med_photogeo_pc": dist_bj_k,
        "sample": samp_k, "train": train_k,
        "spec_bad_frac": star_bad,
    })
    out.to_parquet(args.out, index=False)

    model_out = args.model_out or (os.path.splitext(args.out)[0] + "_model.npz")
    save_model(model_out,
               theta_all=theta_all, stats_all=stats_all,
               theta_A=theta_A, stats_A=stats_A, theta_B=theta_B, stats_B=stats_B,
               good=good, frac_sigma=frac_sigma, tel_masks=tel_masks,
               config={"lam": args.lam, "f_max": 2.0, "bad_frac": args.bad_frac,
                       "snr_min": args.snr_min, "seed": args.seed})
    log(f"wrote {args.out}  ({len(out)} stars)")
    log(f"saved model -> {model_out}")

    # ---- evaluation report on the training subset (cross-validated) ----
    cat = {"plx_a": plx_tr, "err_a": err_tr,
           "plx_sp": plx_sp[train_k], "err_sp": err_sp[train_k],
           "train": np.ones(train_k.sum(), bool), "sample": fold_tr,
           "id": ids_k[train_k]}
    print()
    E.print_report(E.evaluate(cat, label="full-data linear fit"))
    print()
    for f in ("A", "B"):
        E.print_report(E.evaluate(cat, fold=f, label="full-data linear fit"))
        print()


if __name__ == "__main__":
    main()
