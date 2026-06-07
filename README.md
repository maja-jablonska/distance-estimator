# Spectrophotometric Parallax — Baseline Eval Harness

Reproduces and scores Hogg, Eilers & Rix (2018) so a new model (e.g. a
heteroscedastic neural net) is judged on identical footing.

## Files
- `hogg2018.fits` — published Zenodo catalog (record 1468053), 44,784 stars.
  Columns: 2MASS_ID, Gaia_parallax(_err), spec_parallax(_err), training_set, sample.
- `spphot_eval.py`  — metrics (robust scatter, bias, chi2) + catalog loader.
- `spphot_plots.py` — Fig 2, residual histogram, two-model comparison.

## Baseline numbers (verified against the paper)
- N training = 28,226 ; negative Gaia parallaxes = 4.9% (paper: ">2%")
- hi-S/N (Gaia S/N>=20) probe: robust fractional scatter = 9.4%  (paper: <9%)
- bias (median frac resid) = -2.7%
- chi2 mean/median/robust = 2.96 / 0.89 / 1.94
  -> mean >> median == OUTLIER-DRIVEN, exactly as the paper reports.

## How to score your NN
Your model writes a FITS with the four parallax columns (copy Gaia cols
through unchanged; fill spec_parallax/_err with predictions) on the fold it
did NOT train on. Then:

    import spphot_eval as E, spphot_plots as P
    base = E.load_catalog("hogg2018.fits")
    new  = E.load_catalog("my_nn_foldB.fits")
    E.print_report(E.evaluate(new, fold="B", label="NN"))
    P.compare_scatter(base, new, "compare.png")

## What "beating the baseline" means
1. robust scatter < 9.4%  (tighter core)
2. mean chi2 closer to 1   <- the real win: the aleatoric sigma_int head
   should absorb the dusty/crowded outliers that inflate chi2 here.
3. preserve A/B fold independence: score each fold only with the model
   trained on the complementary fold (never leak).

## NOT reproducible from this catalog alone (need extra crossmatched data)
Figs 3/4/5/6 need APOGEE stellar params, G-mag, sky coords, PMs, RVs.
Pull those by 2MASS-ID crossmatch to the Gaia archive on Gadi.
