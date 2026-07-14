# Spectrophotometric distances to luminous giants in the inner Galaxy — methods outline

Full methods backbone of the paper (working title along these lines). Design
decisions and open choices are flagged as they come up. The repo-mapped, actionable
version of this lives in `PLAN.md`; this file is the complete reference outline.

## 0. Scope and design principles

**Goal:** distances to APOGEE luminous giants (0 < log g ≤ 2.2) valid into the
bulge — i.e., robust to A_V ~ 5–20, crowding, non-standard extinction law, and a
metal-rich α-enhanced population — with per-star, believable uncertainties.

**Principles inherited from Hogg+18, kept invariant throughout:**
(i) never cut, invert, or take logs of parallaxes; all distance information enters
through likelihoods; (ii) leave-out training so no star's anchor data touches its
own prediction; (iii) the simplest model that passes the validation gates wins.

**New principles:** (iv) extrapolation behavior is a design requirement, not an
afterthought — the model must degrade linearly, not silently, outside the training
hull; (v) the dust correction must be *certifiable*, via injection tests, not just
assumed learned.

## 1. Data

**Spectra:** APOGEE DR17, aspcapStar, pseudo-continuum-normalized per The Cannon
convention, common rest-frame grid (~7400 pixels). Sample cut on 0 < log g ≤ 2.2
(ASPCAP), same as Hogg+18.

**Photometry, tiered by field density:**
- All sky: Gaia DR3 G, BP, RP; 2MASS J, H, Ks; WISE W1, W2.
- Inner Galaxy (|l| < 10°, |b| < 5°, plus a density criterion): replace 2MASS with
  VVV/VIRAC2 JHKs; drop or down-weight W1/W2 above a crowding threshold.
  Cross-calibration: fit VVV↔2MASS offsets as linear-in-color transforms on overlap
  stars; include a survey-indicator feature so residual system differences are
  absorbable.
- Every band carries a per-star uncertainty and a blend/contamination flag
  (neighbor density within PSF).

**Astrometry:** Gaia DR3 parallaxes with Lindegren+21 zero-point applied per star
(G, color, β); one residual global offset ϖ₀ kept as a free nuisance parameter.
Quality control via RUWE with a crowding-dependent threshold — chosen to be as
permissive as defensible, and explicitly checked to be uncorrelated with true
distance at fixed sky position.

**Anchors (the training-hull extension):**
- Globular clusters: APOGEE members of clusters with Baumgardt–Vasiliev distances,
  emphasizing inner-Galaxy/high-A_V clusters and metal-rich ones; membership by
  RV + proper motion + [Fe/H] consistency.
- Asteroseismology: APOGEE–Kepler/K2 giants with seismic luminosities (inner-disk
  K2 campaigns especially).
- Each anchor contributes a distance-modulus likelihood term, with its own
  systematic error floor and a survey/method indicator.

**External extinction inputs (features, not corrections):** RJCE A_Ks where W2
exists; VVV-based E(J−Ks) maps (Gonzalez+2012 / Surot+2020) in the bulge;
Marshall+2006 3D as a fallback. Carried as a per-star proxy A with an
uncertainty — used by the model (§2) and stratification (§5), never to pre-correct
the target.

## 2. Model

Predicted parallax: ϖ_pred = exp(f(x)), with the feature vector
x = [1, photometry block m, extinction proxy A, spectral block ln f_λ].

**Structure (linear-anchored residual network):**

```
f(x) = θ·x + A·(φ·x_m) + g_NN(x; w)
```

- θ·x — the Hogg+18 linear backbone; L1 on spectral coefficients as before.
- A·(φ·x_m) — bilinear extinction term on the photometric block only
  (φ L1-regularized, initialized 0): captures extinction-coefficient drift with A,
  the dominant known nonlinearity.
- g_NN — small MLP (input: same x, or a PCA/attention-reduced spectral
  representation ~50–200 dims; 2 hidden layers, width ≲ 128; strong weight decay +
  an explicit penalty λ_g‖g‖² pulling it to zero). Captures whatever nonlinearity
  remains.

**Extrapolation guard:** g_NN's output is gated by a training-density score d(x)
(e.g., normalizing-flow log-density or k-NN distance in the reduced feature space):

```
f = θ·x + A(φ·x_m) + σ(d(x))·g_NN(x)
```

Outside the hull the model reverts to the linear+bilinear backbone by construction.
This is principle (iv) made mechanical.

**Interpretability handles retained:** θ_phot projected onto reddening vectors
(R_V ∈ [2.2, 3.3]) certifies the linear backbone's dust behavior; ‖g_NN‖ and its
gate statistics quantify how much nonlinearity the data demanded and where.

## 3. Likelihood and training objective

Heterogeneous sum over data types, all in their natural measurement space:

- **Field stars (parallax space):** robust two-component mixture — with probability
  1−ε, ϖ_Gaia ~ N(ϖ_pred + ϖ₀, σ_n²); with probability ε, an inflated-variance bad
  component. σ_n² = σ_Gaia,n² + (feature-error propagation term) + s²(x), where
  s(x) is a learned per-star scatter head (β-NLL, β ≈ 0.5, as in the
  heteroscedastic pipeline) — intrinsic population scatter and un-modeled crowding
  effects land here.
- **Anchor stars (distance-modulus space):** μ_anchor ~ N(μ_pred, σ_anchor² + s_μ²),
  with μ_pred = −5 log₁₀(ϖ_pred) − 10 + offset conventions; per-anchor-class
  systematic floors.
- **Regularizers:** λ₁‖P_spec θ‖₁ + λ₂‖φ‖₁ + λ_g‖w‖² (+ weight decay inside g).

**Training protocol:** A/B split at the *star* level, with entire clusters assigned
to one side (cluster anchors must not leak). Optimization staged: (1) fit θ alone
on the high-S/N subset (convex-ish warm start, avoiding the exp-underflow pathology
Hogg+18 identify); (2) unfreeze φ, ϖ₀, scatter head; (3) unfreeze g_NN.
Adam or L-BFGS in JAX; hyperparameters (λ's, ε, gate bandwidth) by cross-validated
predictive score on the A/B split, with the *bulge-relevant* metrics of §5 given
veto power, not just global RMS.

## 4. Outputs

Per star: ϖ_sp, σ_ϖ (from feature propagation + scatter head), hull-density score
d(x), gate value, and a distance posterior computed properly (Gaussian likelihood
in ϖ × chosen prior — released as likelihood parameters so users can apply their
own prior, per Hogg+18's closing recommendation). For the orbit-integration use
case downstream: sample distances from the ϖ-space likelihood, never invert.

## 5. Validation gates (each tier of §6 must pass before promotion)

1. **Reddening injection:** artificially redden held-out low-A stars' photometry
   (Fitzpatrick 2019; A_V grid to 10; R_V ∈ {2.5, 3.1}) — slope of Δln ϖ_sp vs
   injected A_V must be consistent with zero, per R_V. This is the headline
   certificate of the dust correction.
2. **Coefficient projection:** θ_phot ⊥ reddening-vector family, quantified
   residual.
3. **Residual regressions (LOCO-style):** normalized (ϖ_sp − ϖ_Gaia) against A_Ks,
   [Fe/H], [α/Fe], T_eff, crowding — jointly, on the disk where Gaia has power;
   framework ported from the existing LOCO diagnostics.
4. **Red clump ruler:** distance histograms along VVV bulge sightlines must
   reproduce RC peak positions, including the X-shape split at |b| ≳ 5°.
5. **Held-out clusters:** anchors excluded from training (rotating), emphasizing
   high-A_V and metal-rich; revisit the M71 anomaly explicitly on DR17/DR3.
6. **External cross-checks:** Mira/RRL PL distances in matched fields;
   StarHorse/astroNN comparisons stratified by A_Ks (agreement not required, but
   divergences must be explainable).
7. **Ablation report:** each tier's marginal effect on gates 1–5; ‖g_NN‖ and
   gate-activation maps published as part of the method's honesty budget.

## 6. Staging (each stage is a controlled ablation)

1. Hogg+18 reproduction on DR17 + DR3 + Lindegren zero-points — baseline.
2. + VVV photometry swap, feature-error propagation, mixture likelihood, relaxed
   astrometric cuts.
3. + Anchors (clusters + seismic) in the likelihood; ϖ₀ nuisance.
4. + Bilinear extinction term + extinction-proxy feature.
5. + Scatter head (β-NLL).
6. + Gated residual NN — *only if* gate 1's injection slope or gate 3's residual
   trends remain significant after stage 4–5.

Stop at the first stage where gates 1, 3, and 4 pass; later stages become appendix
ablations rather than the fiducial model.

## Open decisions to settle early

- The crowding threshold for the photometry tiers (affects sample size vs. purity;
  worth a quick VIRAC2 density study on Gadi first).
- Whether the spectral block feeds g_NN raw or reduced (reduction helps
  overfitting, costs some information — start reduced).
- The anchor systematic floors (cluster distance scale disagreements at the
  ~0.05 mag level set a hard floor on what the anchors can certify).
- The prior family to recommend for the released distance posteriors (bulge users
  will want a density prior; disk users exponential — releasing likelihood
  parameters sidesteps choosing).
