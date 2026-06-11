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
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # for spphot_eval / spphot_plots

LABEL_COLS = ["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
META_COLS  = ["sdss_id", "plx", "e_plx", *LABEL_COLS, "r_med_photogeo"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    log("reading parquet scalar columns (sdss_id, snr, spectrum_flags) ...")
    meta = pq.read_table(parquet_path,
                         columns=["sdss_id", "snr", "spectrum_flags"]).to_pandas()
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
                           batch_rows=20000, fixed_good=None):
    """One streaming pass. Returns:
        X_spec    (n_keep, L) float32  ln(normalized flux) on shared good pixels
        good      (Lfull,)    bool     kept-pixel mask
        star_bad  (n_keep,)   float32  per-star bad-pixel fraction (data-quality flag)
    keep_mask is in PARQUET ROW ORDER (same order as load_metadata's merged frame).
    If fixed_good is given (a saved model's mask), it is used verbatim instead of
    recomputing — so new spectra land on exactly the pixels the model was fit on."""
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

    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows,
                                               columns=["flux", "continuum", "ivar"])):
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

        flux = _list_col_2d(batch.column("flux"), width)[sel]
        cont = _list_col_2d(batch.column("continuum"), width)[sel]
        ivar = _list_col_2d(batch.column("ivar"), width)[sel]

        C = np.where(cont > 0, cont, np.nan)
        f = flux / C                                              # normalized flux
        bad = (~np.isfinite(f) | ~np.isfinite(ivar) | (ivar <= 0)
               | (cont <= 0) | (f <= 0) | (f > f_max))            # ivar==0 is APOGEE's mask sentinel
        f = np.where(bad, 1.0, f)                                 # impute bad -> continuum (ln 0)
        lnfull[out:out + k] = np.log(f).astype(np.float32)
        bad_count += bad.sum(axis=0)
        star_bad[out:out + k] = bad.mean(axis=1)
        out += k

        if bi % 20 == 0:
            log(f"  spectra: {out}/{n_keep} kept rows "
                f"({(time.time()-t0):.0f}s)")

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

    @jax.jit
    def value_and_grad(theta):
        def obj(th):
            m = jnp.exp(jnp.clip(Xj @ th, -30, 30))
            r = (yj - m) / sj
            return 0.5 * invN * jnp.sum(r ** 2) + 0.5 * jnp.sum(regj * th ** 2)
        return jax.value_and_grad(obj)(theta)

    def scipy_obj(theta):
        fv, g = value_and_grad(jnp.asarray(theta))
        return float(fv), np.asarray(g, np.float64)

    res = minimize(scipy_obj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter})
    return res.x, res


def predict(theta, stats, phot, spec, batch=50000):
    """exp(theta . x) in batches so we never standardize the whole 800k at once."""
    out = np.empty(len(phot))
    for i in range(0, len(phot), batch):
        Xb, _ = design(phot[i:i + batch], spec[i:i + batch], stats)
        out[i:i + batch] = np.exp(np.clip(Xb @ theta, -30, 30))
    return out


# ----------------------------------------------------------------------
# model persistence — one self-contained, re-loadable artifact
# ----------------------------------------------------------------------
def save_model(path, *, theta_all, stats_all, theta_A, stats_A, theta_B, stats_B,
               good, frac_sigma, config):
    """Everything needed to predict on NEW stars without refitting: the all-train
    model (theta + standardization), the good-pixel mask, the adopted fractional
    error, the feature order, and the build config. The fold models are kept too
    so the A/B cross-validation is reproducible. Load with load_model()."""
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
    )


def load_model(path):
    """Return a dict with theta_all, stats_all=(mu,sd), good_pixel_mask, frac_sigma,
    f_max, bad_frac, label_cols — the pieces apply_model.py needs."""
    z = np.load(path, allow_pickle=False)
    return {
        "theta_all": z["theta_all"], "stats_all": (z["mu_all"], z["sd_all"]),
        "theta_A": z["theta_A"], "stats_A": (z["mu_A"], z["sd_A"]),
        "theta_B": z["theta_B"], "stats_B": (z["mu_B"], z["sd_B"]),
        "good_pixel_mask": z["good_pixel_mask"], "frac_sigma": float(z["frac_sigma"]),
        "f_max": float(z["f_max"]), "bad_frac": float(z["bad_frac"]),
        "label_cols": [str(c) for c in z["label_cols"]],
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
    ap.add_argument("--snr-min", type=float, default=100.0, help="training S/N cut")
    ap.add_argument("--bad-frac", type=float, default=0.01, help="shared good-pixel threshold")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import pandas as pd

    merged, n_parquet = load_metadata(args.parquet, args.allstar)

    # ---- keep = complete photometry (we build spectra + predict for all of these) ----
    phot_all = merged[LABEL_COLS].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    log(f"keep (complete photometry): {keep.sum()} / {n_parquet}")

    # ---- training set: quality cuts only; NO cut on parallax sign or S/N ----
    plx_all = merged["plx"].to_numpy(float)
    err_all = merged["e_plx"].to_numpy(float)
    snr_ok  = merged["snr"].to_numpy(float)
    flags   = merged["spectrum_flags"].to_numpy()
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
        args.parquet, keep, bad_frac_max=args.bad_frac, batch_rows=args.batch_rows)

    # ---- restrict metadata to kept rows (same order as X_spec) ----
    phot_k = phot_all[keep]
    plx_k, err_k = plx_all[keep], err_all[keep]
    samp_k = sample[keep]
    train_k = train[keep]
    ids_k = merged["sdss_id"].to_numpy()[keep]
    dist_bj_k = merged["r_med_photogeo"].to_numpy(float)[keep]

    # ---- fit three models on the training subset ----
    phot_tr, spec_tr = phot_k[train_k], X_spec[train_k]
    plx_tr, err_tr, fold_tr = plx_k[train_k], err_k[train_k], samp_k[train_k]

    def fit_on(mask, name):
        Xf, st = design(phot_tr[mask], spec_tr[mask])
        th, res = fit_parallax_model(Xf, plx_tr[mask], err_tr[mask], lam=args.lam)
        log(f"  fit {name}: {mask.sum()} stars | converged={res.success} "
            f"iters={res.nit} obj={res.fun:.4f}")
        return th, st

    log("fitting models ...")
    theta_A, stats_A = fit_on(fold_tr == "A", "fold-A")
    theta_B, stats_B = fit_on(fold_tr == "B", "fold-B")
    theta_all, stats_all = fit_on(np.ones(len(plx_tr), bool), "all-train")

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
        "plx": plx_k, "e_plx": err_k,
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
               good=good, frac_sigma=frac_sigma,
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
