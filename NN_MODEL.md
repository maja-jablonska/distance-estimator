# Heteroscedastic neural-net spectrophotometric parallax

`train_nn.py` — a drop-in replacement for the linear Hogg+18 model that predicts a
parallax **and a per-star error** from a star's spectrum + photometry. It reuses
`run_full_gadi.prepare_sample`, so the sample, features, zero-point, pixel masks
and A/B split are **identical** to the linear baseline — only the model in the
middle changes, which is what makes the comparison fair.

---

## Why a neural net (justification)

The linear model (Hogg, Eilers & Rix 2018) predicts `plx = exp(θ·x)` — a single
linear function of the features. It works remarkably well (~9% scatter) because
the sample is restricted to a narrow slice of stellar parameters (`0 < log g <
2.2`), where the luminosity–spectrum relation is close to linear. But it has two
limits this NN is built to remove:

1. **No per-star uncertainty.** The linear model assigns the *same* fractional
   error to every star (`err_sp = frac_sigma · plx_sp`), so plotting its error vs
   parallax is a straight line. That is not physical — a bright, well-measured
   giant and a faint, noisy one get the same quoted error. A learned,
   heteroscedastic `sig_sp` fixes this and is the main motivation.

2. **Limited flexibility.** A linear map cannot capture curvature in the
   spectrum→luminosity relation (metallicity, age, evolutionary-phase effects,
   residual reddening). An MLP can.

Hogg+18 themselves note (their §6) that "a better set of features would do better"
and that deep models "subsume the linear model … and could deliver better — or at
least non-worse — results." The cost they warn about — flexible models can overfit
and extrapolate badly — is exactly why the eval reports an **overfit gap** (below).

---

## How it works

### Features

Each star is a vector `x = [8 photometric magnitudes | ln-flux over the "good"
spectral pixels]`, **standardized** (subtract mean, divide by std) using
statistics computed *only on the training fold*, so nothing leaks from held-out
stars.

### Architecture — `HetMLP`

```
x → [Linear → SiLU → Dropout] × 3   (widths 1024, 512, 256)   = shared body
body → head_mu    → z_mu
body → head_logs  → z_logs
```

A shared trunk feeds **two linear heads**, mapped to physical quantities:

- `plx_sp = exp(z_mu)` — the spectrophotometric parallax. Log-space keeps it
  **positive** (same trick as the linear `exp(θ·x)`).
- `sig_sp = exp(z_logs)` — the per-star error, also positive, clamped to
  `[1e-4, 10]` mas for numerical safety.

### Likelihood (the loss)

The model assumes the Gaia parallax is the spec prediction plus noise, with the
**known Gaia error folded into the variance**:

```
plx_gaia ~ Normal( plx_sp , sig_sp² + e_plx² )
```

so `sig_sp` learns the *intrinsic* spectrophotometric scatter, not Gaia's noise.
The negative-log-likelihood (`het_nll`):

```
NLL = 0.5 · [ (plx − plx_sp)² / var + log(var) ],   var = sig_sp² + e_plx²
```

Two terms in tension: the first rewards a good mean; the second (`log var`)
penalizes just inflating the error. Balancing them is what *calibrates* `sig_sp`.
Everything is in **parallax space, never inverted to distance**, and negative /
low-S/N Gaia parallaxes are kept (the Hogg "no cuts" principle — cutting on
parallax biases the fit).

**beta-NLL (`--beta`, default 0.5).** Plain NLL down-weights high-variance
(distant, noisy) stars, so the mean head *underfits* them and becomes biased. The
beta-NLL of Seitzer et al. (2022) multiplies each star's loss by `var^β`,
restoring the mean's gradient there. `β=0` = plain NLL (biased mean), `β=0.5` =
recommended, `β=1` ≈ MSE. **Trade-off:** higher β fixes the mean bias but tends to
*underestimate* the variance — this is why the default run is well-centered but
~2× overconfident in `err_sp` (see calibration below).

**Student-t loss (`--loss studentt`).** Same two heads, but the residual is
modelled with fat tails (`ν` d.o.f.), so outliers (binaries, bad astrometry)
contribute ~log of their residual instead of its square and don't drag the fit.
Here `sig_sp` is the t *scale*; it is converted to a 1-σ std at output via
`std_factor = √(ν/(ν−2))`.

### Cross-validation (mirrors the linear pipeline)

Three models are trained: `model_A` (fold A only), `model_B` (fold B only),
`model_all` (all training stars). Each star is then predicted by a model that
**never saw it**: A by `model_B`, B by `model_A`, everything else by `model_all`.
This makes each `plx_sp` statistically **independent of its own Gaia parallax**,
which is what lets you (a) score honestly and (b) later combine spec + Gaia by
inverse-variance weighting.

### Output

A parquet with the same schema as the linear model (`plx_sp`, `err_sp`,
`dist_sp_kpc`, the Gaia columns, `sample`, `train`, …) plus a torch checkpoint
holding the weights, standardization stats, and pixel mask — so `apply_nn.py`
reproduces the exact preprocessing on new stars.

---

## Running it

```bash
# single run
qsub train_nn.pbs
qsub -v EPOCHS=100,HIDDEN="2048,1024,512" train_nn.pbs

# robust variant
qsub -v LOSS=studentt,OUT=/scratch/mk27/$USER/spphot_nn_studentt.parquet train_nn.pbs

# beta-NLL sweep (one sample prep, one model per beta) — the calibration<->bias study
qsub -v BETA_SWEEP="0.2,0.3,0.5,0.7" train_nn.pbs
```

Offline Weights & Biases logging is on by default (`WANDB_MODE=offline`); upload
afterwards from the internet-enabled queue with `qsub sync_wandb.pbs`. See the
training header for all `-v` knobs.

---

## How to interpret the results

The eval report (printed by the run, via `spphot_eval.py`) is scored on the
**high-S/N Gaia probe** (S/N ≥ 20), in parallax space, with robust statistics —
the same footing as Hogg+18.

```
=== het-NN (gauss-beta0.5)  (fold: all) ===
  N total / hi-S/N probe : 216175 / 73205
  bias  (median frac resid)  : -1.19 %
  SCATTER (robust frac, <9% = beats Paper I) : 8.26 %
  median quoted spec err frac: 3.56 %
  chi2  mean / median / robust : 4.33 / 1.02 / 2.23
  overfit gap (robust frac scatter, high-S/N probe):
    in-fold (optimistic) : 7.9x %
    held-out (honest)    : 8.26 %
    gap                  : +0.x pp
  error calibration (high-S/N probe, binned by predicted err frac): …
```

| metric | what it means | healthy value |
|---|---|---|
| **SCATTER** (robust frac) | headline accuracy: robust 1.48·MAD of `(plx_sp−plx_a)/plx_a` | `< 9%` beats Paper I |
| **bias** (median frac resid) | systematic offset of `plx_sp` from Gaia | ≈ 0 (±1–2%) |
| **median quoted spec err** | typical `err_sp/plx_sp` the model reports | should ≈ the actual scatter |
| **chi2 median** | dominated by low-S/N stars → tests Gaia's error, not yours | **a trap — ignore as a calibration claim** |
| **chi2 robust** | robust width² of χ over all stars → honest calibration gauge | ≈ 1 if errors honest |
| **chi2 mean** | outlier-driven (Hogg saw this too) | ≫ 1 is expected, not alarming |
| **overfit gap** | held-out minus in-fold scatter | small → not overfitting |

### The reading of the current run

1. **Accuracy is excellent.** 8.26% scatter beats the paper's 9%, bias −1.2% is
   negligible. The point estimates are good.

2. **Errors are ~2× overconfident** — the one open issue. The robust χ² is 2.23
   (should be ~1 → width √2.23 ≈ 1.5× too small), and the median quoted error
   (3.6%) is roughly half the actual scatter (~8%). The **median χ² ≈ 1 is a trap**:
   it is set by the low-S/N majority where Gaia's (honest) error dominates the
   combined variance, so it barely tests `err_sp`. Trust the **robust** χ² and the
   **calibration table**.

3. **Calibration table** (binned by predicted `err_sp/plx_sp`): `obs/pred` and
   `robust_chi` per bin. Flat `obs/pred > 1` across bins → a single scalar recal
   fixes it; a trend → the miscalibration is magnitude-dependent (a *tilt*) and
   needs a magnitude-dependent recal or a lower β.

4. **Overfit gap** — if in-fold ≈ held-out, the MLP is not overfitting and the
   regularization is right; if in-fold ≪ held-out, raise `--dropout` /
   `--weight-decay` and trust only the held-out number.

### Using the parallaxes despite overconfident errors

The point values are fine; **recalibrate the errors before use**:

- Fit the scalar `c` on the held-out fold (`spphot_eval.recalibration_factor`,
  ≈ 1.5) and apply `err_sp *= c` per fold; verify robust χ² → 1 *and* the cluster
  `internal_chi2` → 1 (see `CLUSTER_TEST.md`, a Gaia-free cross-check).
- Or lower β toward 0 to calibrate natively (watch the bias creep back — that's
  the β trade-off).
- Then consume them through a **likelihood**, never naive `1/plx_sp`: treat
  distance as latent, keep the (recalibrated) error, don't cut on S/N. Combining
  with Gaia by inverse-variance weighting is valid because cross-validation makes
  each `plx_sp` independent of its star's Gaia parallax.

### The β sweep

For each β the report prints scatter, bias, robust χ², and the overfit gap. Tabulate
them and pick the β where **robust χ² is closest to 1 while bias stays small and
scatter ≤ 9%** — that is calibration "for free," without post-hoc rescaling.

---

## Related docs / tools

- `CLUSTER_TEST.md` — external cluster validation (Hogg+18 Fig 4) and the
  Gaia-free `internal_chi2` calibration cross-check.
- `spphot_eval.py` — the metrics above: `evaluate`, `calibration_bins`,
  `recalibration_factor`, `overfit_gap`.
- `apply_nn.py` — apply a saved checkpoint to new stars (used by the cluster test).
