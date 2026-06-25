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


def fractional_residuals(plx_sp, plx_a, offset=0.0):
    """(sp - (a+offset))/a. Defined on the high-S/N probe where a is meaningful.

    SIGN CONVENTION (shared by every residual in this module): the residual is
    spec MINUS offset-adjusted Gaia, so +bias means the spec parallax OVER-predicts
    (distances too short). offset is the Gaia zero-point added to the Gaia side, so
    it enters identically here, in chi2_stats, recalibration_factor,
    calibration_bins, bias_bins and zscore.
    """
    return (plx_sp - (plx_a + offset)) / plx_a


def chi2_stats(plx_sp, plx_a, err_a, err_sp, offset=0.0):
    """Per-star chi (sp predicting offset-adjusted Gaia) under combined errors.

    chi_n = (plx_sp - (plx_a + offset)) / sqrt(err_a^2 + err_sp^2)   # spec - adj Gaia

    Returns mean chi2 (should ~1 if errors honest) and a robust analogue.
    Paper I found mean chi2 > 1 (outlier-driven) but robust version ~ ok. The
    het-Gaussian head + scalar recal fixes the ROBUST chi2; only the Student-t
    head genuinely shrinks the tail that keeps mean chi2 > 1 (see recal note).
    """
    chi = (plx_sp - (plx_a + offset)) / np.sqrt(err_a**2 + err_sp**2)
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

    NOTE: targets the ROBUST (MAD) chi width by construction, so it ignores the
    tail -- after recal, robust_chi2 -> 1 but mean_chi2 stays >> 1. Do not credit
    this scalar with any mean-chi2 improvement; that is the Student-t head's job.
    """
    resid = plx_sp - (plx_a + offset)             # spec - adj Gaia (module convention)
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
# bias localization (where does the median frac residual live?)
# ----------------------------------------------------------------------
# The beta-NLL down-weights the mean head's gradient on high-variance stars by
# var**(beta-1), so at LOW beta the mean underfits the faint/scattered regime and
# the global median_frac_resid picks up a net offset. This bins that residual
# along a chosen axis to see whether the bias is FLAT (a global offset correction
# fixes it) or CONCENTRATED at one end (the high-err / faint tail -> de-bias just
# there, or raise beta). Default axis is the predicted err frac, so the table
# lines up row-for-row with calibration_bins(); pass `by`/`by_label` (e.g. the BJ
# distance r_med_photogeo_pc, or a crossmatched Teff/H mag) to bin on anything.
def bias_bins(plx_sp, plx_a, err_a, err_sp, by=None, by_label="pred_err_frac",
              offset=0.0, nbins=8, snr_thresh=20.0):
    probe = hi_snr_mask(plx_a, err_a, snr_thresh)
    axis = (err_sp / plx_sp) if by is None else np.asarray(by, float)
    m = probe & np.isfinite(axis) & (plx_a != 0) & np.isfinite(plx_sp)
    plx_sp, plx_a, axis = plx_sp[m], plx_a[m], axis[m]
    frac = (plx_sp - (plx_a + offset)) / plx_a
    edges = np.quantile(axis, np.linspace(0, 1, nbins + 1))
    edges[-1] = np.inf
    out = []
    for i in range(nbins):
        b = (axis >= edges[i]) & (axis < edges[i + 1])
        if b.sum() < 20:
            continue
        out.append({
            "n":        int(b.sum()),
            "axis_med": float(np.median(axis[b])),
            "bias":     float(np.median(frac[b])),       # median frac resid in bin
            "scatter":  float(robust_scatter(frac[b])),  # robust frac scatter in bin
        })
    return out, by_label


def print_bias_bins(result):
    bins, by_label = result
    print(f"  bias localization (high-S/N probe, binned by {by_label}):")
    print(f"    {by_label[:9]:>9}   bias%   scat%      N")
    for r in bins:
        print(f"    {r['axis_med']:9.3f}  {100*r['bias']:+6.2f}  {100*r['scatter']:5.2f}  {r['n']:6d}")
    print("    (flat bias column = global offset fixes it; trend = bias lives in"
          " one regime -> de-bias there or raise beta)")


# ----------------------------------------------------------------------
# distributional honesty: z-score, coverage, sharpness
# ----------------------------------------------------------------------
# The robust scatter/chi2 above DELIBERATELY ignore the tail (that is why
# mean_chi2 >> robust_chi2). These three look at the whole predictive
# distribution: z exposes the tail's shape, coverage turns it into a number a
# referee reads, and sharpness asks whether the per-star error head is doing any
# real work or whether a single constant error would do as well.
def zscore(plx_sp, plx_a, err_a, err_sp, offset=0.0):
    """Normalized residual z = (plx_sp - (plx_a+offset)) / sqrt(err_sp^2+err_a^2).

    Honest, Gaussian errors -> z ~ N(0,1). One histogram/QQ then shows all three
    failure modes at once: a shifted center (bias), a width != 1 (the same
    over/under-confidence calibration_bins measures) and fat shoulders (the
    outlier tail that drives mean_chi2 >> robust_chi2). Fat shoulders with a
    well-centered, unit-width core is the empirical case for the Student-t head.
    """
    z = (plx_sp - (plx_a + offset)) / np.sqrt(err_sp**2 + err_a**2)
    return z[np.isfinite(z)]


def coverage(plx_sp, plx_a, err_a, err_sp, offset=0.0, levels=(1.0, 2.0, 3.0)):
    """Empirical fraction of stars with Gaia within k*sigma of plx_sp, vs the
    nominal Gaussian 68.3 / 95.4 / 99.7%. emp < nominal -> overconfident."""
    from math import erf, sqrt
    z = np.abs(zscore(plx_sp, plx_a, err_a, err_sp, offset))
    return [{"k": float(k), "emp": float(np.mean(z <= k)),
             "nominal": float(erf(k / sqrt(2.0)))} for k in levels]


def print_coverage(rows):
    print("  coverage (Gaia within k*sigma of plx_sp; honest Gaussian -> nominal):")
    print("     k     empirical  nominal")
    for r in rows:
        flag = "  <- overconfident" if r["emp"] < r["nominal"] - 0.01 else ""
        print(f"    {r['k']:.0f}sig    {100*r['emp']:5.1f}%    {100*r['nominal']:5.1f}%{flag}")


def _spearman(x, y):
    """Spearman rho without scipy: Pearson correlation of the ranks. Ties get
    ordinal (not averaged) ranks -- fine for continuous err_sp/residuals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else np.nan


def sharpness(plx_sp, plx_a, err_a, err_sp, offset=0.0, snr_thresh=20.0):
    """Spearman rho between predicted err_sp and realized |residual| on the
    hi-S/N probe (where err_sp carries the scatter). rho ~ 0 means the
    heteroscedastic head is decorative -- a constant error would score as well;
    a clearly positive rho means beta is buying real per-star uncertainty."""
    probe = hi_snr_mask(plx_a, err_a, snr_thresh)
    resid = np.abs(plx_sp - (plx_a + offset))
    m = probe & np.isfinite(resid) & np.isfinite(err_sp) & (err_sp > 0)
    return _spearman(err_sp[m], resid[m])


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
    frac = fractional_residuals(plx_sp[probe], plx_a[probe], offset)

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
