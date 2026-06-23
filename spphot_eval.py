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
# error calibration (heteroscedastic-NN specific)
# ----------------------------------------------------------------------
# A median chi2 ~ 1 is a TRAP: it is dominated by the low-S/N majority where the
# (honest) Gaia err_a sets the combined error sqrt(err_a^2+err_sp^2), so it
# barely tests err_sp. The honest calibration probes are (a) robust chi2 over
# all stars and (b) the high-S/N probe, where err_a is tiny so err_sp must carry
# the scatter. The two functions below expose whether the quoted err_sp is
# honest and, if not, by how much (and whether the miscalibration is S/N- /
# magnitude-dependent rather than a flat scale error).

def _robust_chi_width(resid, err_a, err_sp, c=1.0):
    """Robust width (1.48*MAD) of chi with err_sp scaled by c, centered on its
    own median. Honest combined error -> width ~ 1."""
    chi = resid / np.sqrt(err_a**2 + (c * err_sp)**2)
    return robust_scatter(chi)


def recalibration_factor(plx_sp, plx_a, err_a, err_sp, offset=0.0, target=1.0):
    """Scalar c such that err_sp -> c*err_sp makes the robust chi width ~ target.

    Scales ONLY the spec error inside the combined denominator, so the fit
    correctly accounts for the (honest) Gaia error: on Gaia-dominated stars c
    barely matters, on spec-dominated stars it does the work. The width is
    monotonic-decreasing in c, so a simple bisection is robust and needs no
    scipy. c > 1 means the NN is overconfident (errors too small).

    Fit this on the HELD-OUT fold and apply (err_sp *= c) before writing the
    catalog. A single scalar only fully fixes a FLAT miscalibration; if
    calibration_bins() shows a tilt, prefer a magnitude-dependent recal or a
    lower beta-NLL.
    """
    resid = plx_a + offset - plx_sp
    m = np.isfinite(resid) & np.isfinite(err_a) & np.isfinite(err_sp) & (err_sp > 0)
    resid, err_a, err_sp = resid[m], err_a[m], err_sp[m]
    lo, hi = 1e-3, 100.0
    for _ in range(60):                       # ~1e-18 precision; monotone -> safe
        mid = 0.5 * (lo + hi)
        if _robust_chi_width(resid, err_a, err_sp, mid) > target:
            lo = mid                          # width too big -> errors too small -> raise c
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibration_bins(plx_sp, plx_a, err_a, err_sp, offset=0.0, nbins=8,
                     snr_thresh=20.0):
    """Per-bin error-calibration check, binned by predicted fractional spec error.

    Binning and the observed-sigma deconvolution are done on the high-S/N probe
    (S/N >= snr_thresh), where err_a is small enough that subtracting it in
    quadrature is meaningful. Each bin reports:
      pred_frac : median predicted err_sp/plx_sp
      obs_frac  : empirical spec scatter, deconvolved -> sqrt(MAD(resid)^2
                  - median(err_a^2)) / median(plx_sp)
      ratio     : obs/pred  (>1 -> OVERCONFIDENT in this bin)
      robust_chi: robust width of chi in the bin (unit-free; ~1 if honest)

    A roughly constant ratio across bins == flat miscalibration (a scalar recal
    fixes it). A ratio that climbs/falls across bins == a tilt (needs a
    magnitude-dependent recal or a lower beta-NLL).
    """
    probe = hi_snr_mask(plx_a, err_a, snr_thresh)
    fp = err_sp / plx_sp
    m = probe & np.isfinite(fp) & (plx_sp > 0) & (err_sp > 0) & np.isfinite(err_a)
    plx_sp, plx_a, err_a, err_sp, fp = (plx_sp[m], plx_a[m], err_a[m],
                                        err_sp[m], fp[m])
    edges = np.quantile(fp, np.linspace(0, 1, nbins + 1))
    edges[-1] = np.inf                        # include the top point
    out = []
    for i in range(nbins):
        b = (fp >= edges[i]) & (fp < edges[i + 1])
        if b.sum() < 20:
            continue
        resid = plx_sp[b] - (plx_a[b] + offset)
        obs_var = robust_scatter(resid)**2 - np.median(err_a[b]**2)
        obs_sigma = np.sqrt(max(obs_var, 0.0))
        med_plx = np.median(plx_sp[b])
        pred_frac = float(np.median(fp[b]))
        obs_frac = float(obs_sigma / med_plx) if med_plx > 0 else np.nan
        out.append({
            "n":         int(b.sum()),
            "pred_frac": pred_frac,
            "obs_frac":  obs_frac,
            "ratio":     float(obs_frac / pred_frac) if pred_frac > 0 else np.nan,
            "robust_chi": float(_robust_chi_width(resid, err_a[b], err_sp[b])),
        })
    return out


def print_calibration(bins, c=None):
    print("  error calibration (high-S/N probe, binned by predicted err frac):")
    print("    pred%   obs%   obs/pred  robustChi    N")
    for r in bins:
        print(f"    {100*r['pred_frac']:5.2f}  {100*r['obs_frac']:5.2f}   "
              f"{r['ratio']:6.2f}     {r['robust_chi']:5.2f}   {r['n']:6d}")
    print("    (obs/pred & robustChi > 1 -> overconfident; flat across rows ="
          " scalar recal ok, trend = needs mag-dependent recal)")
    if c is not None:
        print(f"  global recal factor (err_sp *= c -> robust chi ~ 1): c = {c:.3f}")


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


def calibrate(cat, snr_thresh=20.0, offset=0.0, fold=None, nbins=8):
    """Calibration report mirroring evaluate()'s fold handling.

    Returns (bins, c): the per-bin calibration table and the global recal factor
    fit on the SAME selection. To recalibrate honestly, fit c on the held-out
    fold (e.g. fold='A') and apply err_sp *= c to that fold's catalog rows.
    """
    sel = np.ones(len(cat["plx_a"]), bool)
    if fold is not None:
        sel &= (cat["sample"] == fold)
    plx_a, err_a = cat["plx_a"][sel], cat["err_a"][sel]
    plx_sp, err_sp = cat["plx_sp"][sel], cat["err_sp"][sel]
    bins = calibration_bins(plx_sp, plx_a, err_a, err_sp, offset, nbins, snr_thresh)
    c = recalibration_factor(plx_sp, plx_a, err_a, err_sp, offset)
    return bins, c


def print_report(rep):
    print(f"=== {rep['label']}  (fold: {rep['fold']}) ===")
    print(f"  N total / hi-S/N probe : {rep['n_total']} / {rep['n_hi_snr_probe']}")
    print(f"  bias  (median frac resid)  : {100*rep['median_frac_resid']:+.2f} %")
    print(f"  SCATTER (robust frac, <9% = beats Paper I) : {100*rep['robust_frac_scatter']:.2f} %")
    print(f"  median quoted spec err frac: {100*rep['median_spec_err_frac']:.2f} %")
    print(f"  chi2  mean / median / robust : "
          f"{rep['chi2_mean_chi2']:.2f} / {rep['chi2_median_chi2']:.2f} / {rep['chi2_robust_chi2']:.2f}")
    print(f"    (mean chi2 >> 1 means outlier-driven; honest errors -> ~1)")


def overfit_gap(infold, heldout):
    """Compare robust fractional scatter from IN-FOLD vs HELD-OUT predictions.

    infold/heldout are evaluate() reports built from, respectively, predictions
    where each star was scored by a model that DID train on it, and by one that
    did NOT. The linear L1 model barely overfits; an MLP can. A large gap means
    the held-out (honest) number is the only one to trust and you should raise
    dropout / weight_decay or shrink the net.
    """
    a, b = infold["robust_frac_scatter"], heldout["robust_frac_scatter"]
    return {"in_fold": a, "held_out": b, "gap_pp": 100 * (b - a)}


def print_overfit_gap(infold, heldout):
    g = overfit_gap(infold, heldout)
    verdict = ("ok (not overfitting)" if g["gap_pp"] < 1.0
               else "OVERFIT -> raise dropout/weight_decay or shrink net")
    print("  overfit gap (robust frac scatter, high-S/N probe):")
    print(f"    in-fold (optimistic) : {100*g['in_fold']:.2f} %")
    print(f"    held-out (honest)    : {100*g['held_out']:.2f} %")
    print(f"    gap                  : {g['gap_pp']:+.2f} pp  ({verdict})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "hogg2018.fits"
    cat = load_catalog(path)
    print_report(evaluate(cat, label="Hogg+18 baseline"))
    print_calibration(*calibrate(cat))
    print()
    for f in ("A", "B"):
        print_report(evaluate(cat, fold=f, label="Hogg+18 baseline"))
        print_calibration(*calibrate(cat, fold=f))
        print()
