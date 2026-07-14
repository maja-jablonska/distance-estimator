# CODE_MAP — paper plan (OUTLINE.md) → code

The repo-mapped, actionable version of the plan is `PLAN.md`; the full methods
outline is `OUTLINE.md`. This file maps each outline item to where it lives in
the code and whether it exists yet. Library code is the `spphot/` package; the
root `.py` files are thin CLI drivers (the PBS entry points, same names/flags
as before the package split).

## §0–§1 Principles + data

| outline item | code | status |
|---|---|---|
| never cut/invert/log parallaxes; all info via likelihoods | `spphot/data.py` `prepare_sample` (no plx cuts in `train`); `spphot/eval.py` (parallax-space scoring) | done |
| leave-out training (A/B folds; anchors fold-reassigned) | `spphot/data.py` `prepare_sample` (seeded split); `spphot/v2.py` `load_anchors` | done |
| DR17 aspcapStar spectra, ln-flux, common grid | `spphot/data.py` `build_lnflux_streaming`; masks from `build_pixel_mask.py` | done |
| photometry tiers / VVV-VIRAC2 swap in the inner Galaxy | `spphot/datasets.py` — `DatasetSpec` + `AuxPhot` + `survey_indicator`; `--dataset`/`--aux-phot` on every driver | **seam scaffolded; no VVV spec yet** (needs the VIRAC2 crossmatch + crowding study) |
| per-band uncertainties + blend flags | — | missing (stage-2 work) |
| Lindegren zero-point per star + ϖ₀ nuisance | zeropoint column via `build_pixel_mask.py`; `plx − zeropoint` in `prepare_sample`; ϖ₀ = `pi0` in `spphot/v2.py` | done |
| RUWE quality control (crowding-dependent) | — | missing (open decision) |
| anchors: clusters (Baumgardt–Vasiliev) + seismic | likelihood: `spphot/v2.py` `anchor_nll`, `load_anchors` (`--anchors` table `sdss_id, mu, mu_err, group`); memberships: `spphot/clusters.py` `load_vasiliev_baumgardt`, `clusters/catalogues/` | **likelihood done; anchor table construction not started** |
| extinction proxy A (RJCE; VVV E(J−Ks) maps later) | `spphot/v2.py` `rjce_aks` + `DatasetSpec.rjce_pair` | RJCE done; external maps missing |

## §2 Model

| outline item | code | status |
|---|---|---|
| linear backbone θ·x (Hogg+18, L1/L2 on spec) | `spphot/linear.py` (`design`, `_gn_fit`, `fit_parallax_model`) — the stage-1 baseline; `spphot/v2.py` `theta` (L1) | done — **9.76% scatter / −1.09% bias** |
| bilinear extinction term A·(φ·x_m) | `spphot/v2.py` `forward` (`z_ext`, L1 on φ, init 0) | done (implemented; unvalidated) |
| gated residual net g_NN + hull-density d(x) | `spphot/v2.py` `fit_pca`, `forward` (PCA-whitened `d`, sigmoid gate, zero-init `g`) | done (implemented; unvalidated) |
| feature layout swappability | `spphot/datasets.py` `FeatureLayout`, threaded through `spphot/v2.py` | done |
| interpretability: θ_phot ⊥ reddening vectors; ‖g‖ maps | honesty-budget print in `train_v2.py`; coefficient projection | budget print done; projection **missing (gate 2)** |

## §3 Likelihood + training

| outline item | code | status |
|---|---|---|
| field mixture likelihood (1−ε)N + εN(inflated) | `spphot/v2.py` `field_nll` | done |
| scatter head s(x), β-NLL | `spphot/v2.py` (`s` MLP); prototype `spphot/nn.py` `het_nll` — 8.26% scatter, errors ~2× overconfident (NN_MODEL.md) | done; calibration open issue |
| Student-t robust variant | `spphot/nn.py` `studentt_nll` | done |
| anchors in distance-modulus space | `spphot/v2.py` `anchor_nll` (linear in z, overflow-free) | done |
| staged unfreezing (warm start → mixture → residual net) | `spphot/v2.py` `trainable_mask`, `make_step`; `train_v2.py --stages` | done (**stage-1 smoke vs 9.76% baseline on Gadi still pending**) |
| regularizers λ₁‖θ_spec‖₁ + λ₂‖φ‖₁ + λ_g‖g‖² | `spphot/v2.py` `penalties` | done |

## §4 Outputs

| outline item | code | status |
|---|---|---|
| ϖ_sp, σ_ϖ, hull score, gate per star | `train_v2.py` output parquet (`plx_sp, err_sp, hull_d, gate, a_ks_rjce`) | done |
| release likelihood parameters (user-chosen prior) | output schema carries (ϖ_sp, σ) — release format TBD | partial |

## §5 Validation gates

| gate | code | status |
|---|---|---|
| 1. reddening injection (headline certificate) | `injection_test.py` | **missing — next artifact** |
| 2. coefficient projection θ_phot ⊥ reddening vectors | — | missing |
| 3. LOCO residual regressions | `spphot/eval.py` `bias_bins` is the building block | partial |
| 4. red-clump ruler along VVV sightlines | — | missing (blocked on VVV data) |
| 5. held-out clusters + internal χ² | `spphot/clusters.py`; drivers `run_clusters.py`, `run_cluster_test.py`, `prepare_cluster_spectra.py`, `apply_nn.py`; see CLUSTER_TEST.md | done (infra + V&B catalogues in `clusters/`) |
| 6. external cross-checks (Mira/RRL, StarHorse/astroNN) | — | missing |
| 7. ablation report / honesty budget | `train_v2.py` honesty-budget print; `spphot/eval.py` `overfit_gap` | partial |

## §6 Staging

| stage | driver | status |
|---|---|---|
| 1. Hogg+18 reproduction | `run_full_gadi.py` (+ `run_full_gadi.pbs`) | **done: 9.76% / −1.09%** |
| 2. VVV swap, error propagation, mixture, relaxed cuts | mixture in `train_v2.py`; VVV seam in `spphot/datasets.py` | partial |
| 3. anchors + ϖ₀ | `train_v2.py --anchors` | code done, table missing |
| 4. bilinear extinction + proxy | `train_v2.py` (RJCE built-in) | done |
| 5. scatter head (β-NLL) | `train_nn.py` (prototype), `train_v2.py` (s head) | done |
| 6. gated residual NN | `train_v2.py --stages 3` | done (run only if gates 1/3 fail after 4–5) |

## Entry points (PBS jobs invoke these by name from the repo root)

| script | job | uses |
|---|---|---|
| `run_full_gadi.py` | `run_full_gadi.pbs` | `spphot.data` + `spphot.linear` + `spphot.eval` |
| `train_nn.py` | `train_nn.pbs` | `spphot.data` + `spphot.nn` + `spphot.eval` |
| `train_v2.py` | `train_v2.pbs` | `spphot.data` + `spphot.v2` + `spphot.eval` |
| `apply_nn.py` | `apply_nn.pbs` | `spphot.nn` (`load_bundle`/`apply_to_parquet`) |
| `run_cluster_test.py` | (called by apply_nn.pbs) | `spphot.clusters` |
| `run_clusters.py` | `run_clusters.pbs` | prep + apply + cluster test in one process |
| `prepare_cluster_spectra.py` | `prepare_cluster_spectra.pbs` | standalone (stdlib+numpy) |
| `build_pixel_mask.py` | `build_pixel_mask.pbs` | standalone; produces lit masks + zeropoint |

`spphot_eval.py`, `spphot_plots.py`, `cluster_test.py` at the root are
re-export shims kept for notebooks and docs; new code imports `spphot.eval`,
`spphot.plots`, `spphot.clusters`.

## Tests

`bash tests/smoke_test.sh` (any laptop, <2 min): generates a synthetic fixture
(`tests/make_fixture.py`), runs the whole chain (linear → NN → v2 stage-1 →
apply → cluster test → cluster prep), asserts invariants (`tests/checks.py`),
and prints value digests — compare digests across commits to prove a refactor
is behavior-preserving.

## legacy/

Superseded, not wired into anything: `apply_model.py` (linear-npz apply),
`build_from_parquet.py` (non-streaming builder), `check_pixel_flags.py`
(one-off diagnostic), `assemble_features.py` (+`.pbs`, DR14 web-ETL). See
`legacy/README.md`.
