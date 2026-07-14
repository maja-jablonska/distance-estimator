# Spectrophotometric distances to luminous giants in the inner Galaxy — working plan

This is the methods backbone of the paper, mapped onto what already exists in this
repo (the per-item plan→code table with status lives in `CODE_MAP.md`; the code
itself is organized as the `spphot/` package with thin root drivers). Two invariants inherited from Hogg+18: never cut/invert/log parallaxes (all
distance information enters through likelihoods), and leave-out training so no
star's anchor data touches its own prediction. Two new principles: extrapolation
must degrade *linearly, not silently* outside the training hull (made mechanical by
the density gate in `train_v2.py`), and the dust correction must be *certified* by
injection tests, not assumed learned.

## Staging (each stage is a controlled ablation; stop when gates 1/3/4 pass)

| stage | content | status |
|---|---|---|
| 1 | Hogg+18 reproduction, DR17 + Gaia DR3 + Lindegren zero-points | **DONE** — `run_full_gadi.py`, 9.76% scatter / −1.09% bias (`--clip-sigma 4`) |
| 2 | VVV/VIRAC2 photometry swap in the inner Galaxy, feature-error propagation, mixture likelihood, relaxed astrometric cuts | mixture likelihood: **in `train_v2.py`**; VVV swap + error propagation: not started (needs VIRAC2 pull + crowding-threshold study on Gadi) |
| 3 | Anchors (clusters + seismic) in the likelihood; ϖ₀ nuisance | **in `train_v2.py`** (`--anchors` table; cluster infra in `clusters/`, `prepare_cluster_spectra.py`); anchor *table construction* (Baumgardt–Vasiliev μ + memberships, APOKASC/K2 seismic μ) not started |
| 4 | Bilinear extinction term + extinction-proxy feature | **in `train_v2.py`** (RJCE A_Ks from H−W2 built in; external VVV E(J−Ks) maps later) |
| 5 | Heteroscedastic scatter head (β-NLL) | prototyped in `train_nn.py` (8.26% scatter, errors ~2× overconfident — see `NN_MODEL.md`); carried into `train_v2.py` as the s(x) head |
| 6 | Gated residual NN — only if gate 1 slope / gate 3 trends survive stages 4–5 | **in `train_v2.py`** (stage-3 unfreeze; hull-density gate) |

`train_v2.py` implements stages 2(likelihood)+3+4+5+6 as *one* model with staged
unfreezing, so each stage boundary is a checkpointed ablation:
`--stages 1` = linear backbone only ≈ the `run_full_gadi` baseline;
`--stages 2` = + mixture, ϖ₀, scatter head, bilinear extinction;
`--stages 3` = + gated residual NN.

## Model (train_v2.py)

```
x       = standardized [phot(8: g,bp,rp,j,h,k,w1,w2) | A_rjce | ln-flux]
z_lin   = θ·x + b                        linear backbone (L1 on spectral θ)
z_ext   = A_std · (φ·x_phot)             bilinear extinction drift (L1 on φ, init 0)
x_red   = [phot | A | PCA_k(spec)]       reduced rep for the gate + residual net
d(x)    = mean squared whitened PCA coordinate  (≈1 inside the training hull)
gate    = σ((τ − d)/w)                   → 0 outside the hull, by construction
g, s    = small MLPs on x_red            residual (zero-init last layer) + log-scatter
ϖ_sp    = exp(z_lin + z_ext + gate·g)
```

Likelihood (heterogeneous sum, natural measurement spaces):
- field stars, parallax space: two-component mixture
  `(1−ε)·N(ϖ; ϖ_sp+ϖ₀, e² + s(x)²) + ε·N(ϖ; ϖ_sp+ϖ₀, e² + s(x)² + σ_bad²)`,
  optional β-NLL weighting (β=0.5 default, per the `train_nn.py` calibration study);
- anchors, distance-modulus space: `μ_anchor ~ N(10 − 5z/ln10, σ_anchor² + floor²)`
  with a per-class systematic floor (default 0.05 mag — the cluster-scale
  disagreement level, a hard floor on what anchors can certify);
- penalties: λ₁‖θ_spec‖₁ + λ₂‖φ‖₁ + λ_g‖g‖² (+ weight decay on g).

Anchor stars get their A/B fold reassigned so an entire cluster lives on one side
(no cluster-level leakage through the fold models).

## Validation gates (promotion criteria, in order of construction)

1. **Reddening injection** *(next artifact — `injection_test.py`)*: redden held-out
   low-A stars' photometry with Fitzpatrick 2019 (A_V grid → 10, R_V ∈ {2.5, 3.1};
   use the `dust_extinction` package for coefficients — do not hand-code them);
   slope of Δln ϖ_sp vs injected A_V must be consistent with 0 per R_V. Run it on
   the *existing linear baseline first* — if stage 1 already passes, later stages
   are appendix material. This is the headline certificate.
2. **Coefficient projection**: θ_phot ⊥ reddening-vector family (R_V ∈ [2.2, 3.3]);
   cheap, reads the `train_v2` checkpoint.
3. **Residual regressions (LOCO-style)**: normalized (ϖ_sp − ϖ_Gaia) vs A_Ks,
   [Fe/H], [α/Fe], T_eff, crowding, jointly, on the disk.
4. **Red-clump ruler**: distance histograms along VVV bulge sightlines must show RC
   peaks + the X-shape split at |b| ≳ 5°.
5. **Held-out clusters**: rotating exclusion, emphasis on high-A_V + metal-rich;
   revisit the M71 anomaly on DR17/DR3. Infra: `cluster_test.py`, `run_clusters.py`,
   `CLUSTER_TEST.md` (internal_chi2 is the Gaia-free calibration cross-check).
6. **External**: Mira/RRL PL distances in matched fields; StarHorse/astroNN
   comparisons stratified by A_Ks.
7. **Ablation report**: marginal effect of each stage on gates 1–5; publish ‖g‖ and
   gate-activation maps (the honesty budget).

## Open decisions

- **Crowding threshold** for the photometry tiers → quick VIRAC2 density study on
  Gadi before committing.
- **VVV↔2MASS cross-calibration**: linear-in-color transforms on overlap stars +
  a survey-indicator feature.
- **Anchor floors**: start at 0.05 mag per class; revisit against Baumgardt–Vasiliev
  vs. Harris scale disagreements.
- **Released posterior prior family**: release likelihood parameters (ϖ_sp, σ) and
  let users choose the prior (Hogg+18's closing recommendation).
- **RUWE threshold**: crowding-dependent, as permissive as defensible; must be
  checked uncorrelated with true distance at fixed sky position.

## Immediate next actions

1. Smoke-test `train_v2.py --stages 1` on Gadi (should reproduce ≈ the linear
   baseline; if not, the plumbing is wrong — fix before adding anything).
2. `--stages 2` and `--stages 3` runs; compare scatter/bias/robust-χ² and the
   overfit gap against 9.76% (linear) and 8.26% (het-NN).
3. Write `injection_test.py` (gate 1) and run it on the stage-1 output.
4. Build the anchor table: Baumgardt–Vasiliev cluster μ for the `clusters/` sample
   + APOKASC-3/K2 seismic μ, columns `sdss_id, mu, mu_err, group`.
5. VIRAC2 crowding study → decide the photometry-tier threshold (stage 2 proper).
