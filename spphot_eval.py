"""
spphot_eval.py — evaluation harness for spectrophotometric parallax models.

Defines the baseline metrics from Hogg, Eilers & Rix (2018) so that any new
model (e.g. a heteroscedastic neural net) is scored on EXACTLY the same
footing. The scoring functions take only parallax-space quantities, so they
work identically on:
  - the published Zenodo catalog (the baseline), and
  - your NN outputs (spec_parallax, spec_parallax_err) on the held-out fold.

Key methodological commitments preserved from Paper I:
  * never cut on parallax or parallax S/N (cuts bias the metric itself)
  * compare in PARALLAX space, never invert to distance for scoring
  * report ROBUST scatter (1.48*MAD), because the chi2 is outlier-driven
  * the high-S/N Gaia subset is an *evaluation probe only* — it plays no
    role in training and is not a model input
"""
from __future__ import annotations
import numpy as np
from astropy.io import fits


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_catalog(path):
    """Load a spphot catalog FITS into a dict of plain numpy arrays.

    Required columns: Gaia_parallax, Gaia_parallax_err,
                      spec_parallax, spec_parallax_err.
    Optional: training_set (1/0), sample ('A'/'B'), 2MASS_ID.
    Your NN catalog only needs to populate spec_parallax[*] and the two
    Gaia columns (copy them through unchanged from the input).
    """
    d = fits.open(path)[1].data
    out = {
        "plx_a":  np.asarray(d["Gaia_parallax"], float),
        "err_a":  np.asarray(d["Gaia_parallax_err"], float),
        "plx_sp": np.asarray(d["spec_parallax"], float),
        "err_sp": np.asarray(d["spec_parallax_err"], float),
    }
    cols = d.columns.names
    out["train"] = (np.asarray(d["training_set"]).astype(bool)
                    if "training_set" in cols else np.ones(len(out["plx_a"]), bool))
    out["sample"] = (np.array([str(s).strip() for s in d["sample"]])
                     if "sample" in cols else np.full(len(out["plx_a"]), "?"))
    out["id"] = (np.array([str(s).strip() for s in d["2MASS_ID"]])
                 if "2MASS_ID" in cols else np.arange(len(out["plx_a"])).astype(str))
    return out


# ----------------------------------------------------------------------
# core metrics
# ----------------------------------------------------------------------
def robust_scatter(x):
    """1.48 * MAD — matches Paper I's robust scatter (insensitive to outliers)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.48 * np.median(np.abs(x - np.median(x)))


def fractional_residuals(plx_sp, plx_a):
    """(sp - a)/a. Defined on the high-S/N probe where a is meaningful."""
    return (plx_sp - plx_a) / plx_a


def chi2_stats(plx_sp, plx_a, err_a, err_sp, offset=0.0):
    """Per-star chi (sp predicting offset-adjusted Gaia) under combined errors.

    chi_n = (plx_a + offset - plx_sp) / sqrt(err_a^2 + err_sp^2)

    Returns mean chi2 (should ~1 if errors honest) and a robust analogue.
    Paper I found mean chi2 > 1 (outlier-driven) but robust version ~ ok.
    A good heteroscedastic NN should push BOTH toward 1.
    """
    chi = (plx_a + offset - plx_sp) / np.sqrt(err_a**2 + err_sp**2)
    chi = chi[np.isfinite(chi)]
    return {
        "mean_chi2":   float(np.mean(chi**2)),
        "median_chi2": float(np.median(chi**2)),
        "robust_chi2": float((1.48 * np.median(np.abs(chi - np.median(chi))))**2),
        "n":           int(chi.size),
    }


def hi_snr_mask(plx_a, err_a, thresh=20.0):
    """Gaia S/N >= thresh. Evaluation probe only — NOT a training cut."""
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = plx_a / err_a
    return np.isfinite(snr) & (snr >= thresh)


# ----------------------------------------------------------------------
# top-level report
# ----------------------------------------------------------------------
def evaluate(cat, snr_thresh=20.0, offset=0.0, fold=None, label="model"):
    """Full metric report on a loaded catalog dict.

    fold : None  -> use all stars
           'A'/'B' -> restrict to that Parent-Sample split (for fair
                      cross-validated comparison: score each NN only on the
                      fold it did NOT train on).
    """
    sel = np.ones(len(cat["plx_a"]), bool)
    if fold is not None:
        sel &= (cat["sample"] == fold)

    plx_a, err_a = cat["plx_a"][sel], cat["err_a"][sel]
    plx_sp, err_sp = cat["plx_sp"][sel], cat["err_sp"][sel]

    probe = hi_snr_mask(plx_a, err_a, snr_thresh)
    frac = fractional_residuals(plx_sp[probe], plx_a[probe])

    rep = {
        "label": label,
        "fold": fold or "all",
        "n_total": int(sel.sum()),
        "n_hi_snr_probe": int(probe.sum()),
        "median_frac_resid": float(np.median(frac)),   # bias
        "robust_frac_scatter": float(robust_scatter(frac)),  # the headline %
        "median_spec_err_frac": float(np.median(err_sp / plx_sp)),
    }
    rep.update({f"chi2_{k}": v for k, v in
                chi2_stats(plx_sp, plx_a, err_a, err_sp, offset).items()})
    return rep


def print_report(rep):
    print(f"=== {rep['label']}  (fold: {rep['fold']}) ===")
    print(f"  N total / hi-S/N probe : {rep['n_total']} / {rep['n_hi_snr_probe']}")
    print(f"  bias  (median frac resid)  : {100*rep['median_frac_resid']:+.2f} %")
    print(f"  SCATTER (robust frac, <9% = beats Paper I) : {100*rep['robust_frac_scatter']:.2f} %")
    print(f"  median quoted spec err frac: {100*rep['median_spec_err_frac']:.2f} %")
    print(f"  chi2  mean / median / robust : "
          f"{rep['chi2_mean_chi2']:.2f} / {rep['chi2_median_chi2']:.2f} / {rep['chi2_robust_chi2']:.2f}")
    print(f"    (mean chi2 >> 1 means outlier-driven; honest errors -> ~1)")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "hogg2018.fits"
    cat = load_catalog(path)
    print_report(evaluate(cat, label="Hogg+18 baseline"))
    print()
    for f in ("A", "B"):
        print_report(evaluate(cat, fold=f, label="Hogg+18 baseline"))
        print()
