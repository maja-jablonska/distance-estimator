# spphot — spectrophotometric distances to APOGEE luminous giants

Distances to APOGEE luminous giants (0 < log g ≤ 2.2) valid into the bulge,
following Hogg, Eilers & Rix (2018) and extending it for the inner Galaxy.
Method docs: `OUTLINE.md` (full plan) → `PLAN.md` (repo-mapped staging) →
`CODE_MAP.md` (plan → code, with status). Model-specific: `NN_MODEL.md`
(het-NN + calibration), `CLUSTER_TEST.md` (cluster validation).

## Layout

```
spphot/            the package: datasets (swap seam), data, linear, nn, v2,
                   eval, plots, clusters   (see spphot/__init__.py)
*.py               entry-point drivers the PBS jobs invoke by name
                   (+ spphot_eval/spphot_plots/cluster_test re-export shims)
*.pbs              Gadi batch jobs
tests/             synthetic fixture + end-to-end smoke test
notebooks/         analysis notebooks (first cell bootstraps sys.path)
legacy/            superseded scripts (see legacy/README.md)
```

No installation needed on Gadi (jobs run from the repo dir); locally,
`pip install -e .` puts `spphot` on any kernel's path.

**After `git pull` on Gadi:** `find . -name __pycache__ -prune -exec rm -rf {} +`
once — stale bytecode from the pre-package layout can shadow the new modules.

## Datasets

Photometry bands are defined in `spphot/datasets.py` (`REGISTRY`); every
driver takes `--dataset` (default `dr17` = Gaia DR3 + 2MASS + WISE) and
`--aux-phot <parquet>` for auxiliary photometry (the upcoming VVV/VIRAC2
swap). Checkpoints record their dataset; apply-time feature assembly always
matches training, including for pre-dataset checkpoints.

## Smoke test

```bash
bash tests/smoke_test.sh        # <2 min on a laptop; PY=<python> to override
```

Runs the whole chain on a synthetic fixture and prints value digests — after
any refactor, identical digests prove behavior is unchanged.

## Baseline (Hogg+18 reproduction, verified)

- Zenodo catalog (`hogg2018.fits`): hi-S/N robust scatter 9.4%, bias −2.7%,
  χ² mean/median/robust 2.96 / 0.89 / 1.94 (outlier-driven, as the paper reports).
- Our DR17+DR3 rerun (`run_full_gadi.py --clip-sigma 4`): **9.76% scatter,
  −1.09% bias** — the stage-1 baseline every later stage must beat (PLAN.md).

## Scoring a new model

Score on the fold the model did NOT train on, in parallax space:

```python
import spphot.eval as E, spphot.plots as P
base = E.load_catalog("hogg2018.fits")
new  = E.load_catalog("my_model_foldB.fits")
E.print_report(E.evaluate(new, fold="B", label="NN"))
P.compare_scatter(base, new, "compare.png")
```

Beating the baseline means: robust scatter below it, χ² near 1 (honest
errors — see the calibration machinery in `spphot.eval` and the Gaia-free
`internal_chi2` in `CLUSTER_TEST.md`), and A/B independence preserved.
