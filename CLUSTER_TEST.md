# Cluster validation of the spectrophotometric parallaxes

The strongest external check on the spec parallaxes, following **Hogg, Eilers &
Rix (2018), Fig. 4**: every member of a cluster sits at one true distance, so its
spectrophotometric parallaxes should (a) form a **tight clump**, (b) have an
inverse-variance-weighted mean that **agrees with the Gaia cluster mean**, and
(c) scatter member-to-member by no more than their **quoted errors** — a check
that uses *no* Gaia parallax at all.

Inputs: one spectra parquet **per cluster** (filename = cluster name), already
crossmatched to astra, plus the shared `astraAllStarASPCAP-0.8.0.fits` (Gaia plx
+ photometry for everyone) and a trained NN checkpoint (`*_model.pt` from
`train_nn.py`).

---

## How to run

### On Gadi (one job does everything)

```bash
git pull
qsub -v CLUSTER_DIR=/scratch/mk27/$USER/clusters,\
MODEL=/scratch/mk27/$USER/spphot_nn_results_model.pt apply_nn.pbs
```

For each `<cluster>.parquet` in `CLUSTER_DIR` the job runs `apply_nn.py` to infer
`plx_sp`/`err_sp` (reusing the exact training features + saved pixel mask), writes
`<cluster>_nn.parquet` into `OUT_DIR` (default `/scratch/mk27/$USER/cluster_nn`),
then prints the cluster test on the concatenation.

Useful `-v` overrides:

| var | meaning | default |
|---|---|---|
| `CLUSTER_DIR` | directory of per-cluster spectra parquets | `${DATA}/clusters` |
| `MODEL` | NN checkpoint to apply | `…/spphot_nn_results_model.pt` |
| `OUT_DIR` | where applied catalogs are written | `…/cluster_nn` |
| `PIXEL_MASK_DIR` | **must match training** | `${SRC}/pixel_mask` |
| `ERR_SCALE` | multiply `err_sp` before scoring (preview the recal factor) | `1.0` |
| `OFFSET` | Gaia zero-point added to `plx` (only if parquets lack `zeropoint`) | `0.0` |
| `PLX_G` | Gaia column to compare against: `plx` (corrected) or `plx_raw` | `plx` |

### Manually / step by step

```bash
# 1) infer per cluster (loop yourself, or let the .pbs do it)
python apply_nn.py --model spphot_nn_results_model.pt \
    --parquet clusters/NGC6791.parquet --allstar astraAllStarASPCAP-0.8.0.fits \
    --pixel-mask-dir <dir> --cluster-name NGC6791 --out cluster_nn/NGC6791_nn.parquet

# 2) run the test on all applied catalogs
python run_cluster_test.py cluster_nn/*_nn.parquet

# preview whether the recalibration factor fixes the errors (cluster-side):
python run_cluster_test.py cluster_nn/*_nn.parquet --err-scale 1.49
```

### In a notebook

```python
import pandas as pd, spphot.clusters as C   # `import cluster_test as C` still works (shim)
df = pd.concat([pd.read_parquet(f) for f in glob.glob("cluster_nn/*_nn.parquet")],
               ignore_index=True)
members = C.members_from_labels(df, "cluster")        # no crosswalk: label is in df
C.print_cluster_test(C.run_cluster_test(df, members, offset=0.0))
```

---

## How to read the output

```
cluster     N  sp_mean   gaia  Δsp/gaia    σ  tight%  intχ²  Δlit%
M67        60    1.174  1.179    -0.4%  -0.6   6.8%    2.7   +0.4
NGC6791    40    0.228  0.243    -6.0%  -1.3   7.7%    3.7   -0.7
```

| column | meaning | healthy value |
|---|---|---|
| `N` | matched members scored | — |
| `sp_mean` | IVW spectrophotometric parallax of the cluster (mas) | — |
| `gaia` | IVW Gaia parallax of the same members (mas) | — |
| `Δsp/gaia` | fractional offset of the two means → **bias** | ≈ 0 |
| `σ` | that offset in sigma → is the bias significant? | **\|σ\| ≲ 2–3** |
| `tight%` | member robust scatter ÷ mean → real distance spread | small, comparable to the global ~8–10% |
| `intχ²` | member spread ÷ quoted `err_sp` → **Gaia-free calibration** | **≈ 1** if errors are honest |
| `Δlit%` | offset vs the (approximate) literature parallax | small (sanity only) |

### The three questions it answers

1. **Are the parallaxes accurate?** Look at `Δsp/gaia` and `σ`. Near zero with
   `|σ| ≲ 2–3` means the spec mean agrees with Gaia for that cluster — the Fig-4
   "two vertical lines coincide" result. A large `|σ|` for one cluster flags a
   population (metallicity/age/extinction) the model handles poorly; Hogg+18 saw
   this for M71.

2. **Are the errors honest?** `intχ²` is the key, **independent** check. Because
   members share one true distance, their `plx_sp` should scatter about the
   cluster mean by exactly `err_sp`. So
   - `intχ² ≈ 1` → calibrated,
   - `intχ² ≈ 4` → ~2× overconfident, and **√intχ² is the recalibration factor**.

   This corroborates the high-S/N-probe finding from `spphot_eval.calibration_bins`
   *without using any Gaia parallax*. Two independent methods agreeing is the
   point. Re-run with `--err-scale <c>` and `intχ²` should fall to ~1.

3. **How clumpy are the distances?** `tight%`. This is the physical spread of
   member distances; it should be small and not much larger than the global
   scatter. A cluster much wider than the others usually means member
   contamination, not model error.

### Worked reading of the example above

- **M67**: `Δsp/gaia −0.4%`, `σ −0.6` → unbiased and consistent with Gaia. Good.
- **NGC6791**: `Δsp/gaia −6%` but `σ −1.3` → the 6% offset is **not significant**
  (Gaia's own mean is uncertain at this distance), so it still passes.
- Both: `intχ² ≈ 3` → errors ~√3 ≈ 1.7× too small, matching the global
  overconfidence. Running `--err-scale 1.49` pulls `intχ²` toward 1.

---

## Gotchas

- **Zero-point.** If a cluster parquet has no `zeropoint` column, `apply_nn` logs
  it and writes `plx = plx_raw`; then score with `-v OFFSET=0.048,PLX_G=plx_raw`
  so the Gaia comparison is fair. With `zeropoint` present, keep `OFFSET=0.0`.
- **Feature dim.** `apply_nn` asserts the cluster features match the checkpoint's
  `d_in`. If it trips, `PIXEL_MASK_DIR` differs from training — point it at the
  same directory used to train.
- **Literature values** in `cluster_test.LIT_PLX_MAS` are approximate placeholders
  for the `Δlit%` sanity column only — replace with Cantat-Gaudin (2020) /
  Baumgardt & Vasiliev (2021) before quoting them anywhere.

## Related tools

- `spphot_eval.py` — headline metrics (scatter, bias, χ²), per-bin error
  **calibration**, and the **overfit gap** (in-fold vs held-out scatter).
- `cluster_test.py` — the metrics used here; also has `add_crosswalk` /
  `match_membership` for the alternative route (external member catalog matched by
  Gaia source id) if you ever score clusters you did *not* pre-crossmatch.
