#!/usr/bin/env bash
# smoke_test.sh — run the whole pipeline end-to-end on the synthetic fixture.
# Fast (<2 min on a laptop CPU) and deterministic: tests/checks.py digest prints
# value hashes to compare across refactor commits.
#
#   bash tests/smoke_test.sh            # full run
#   PY=python3 bash tests/smoke_test.sh # explicit interpreter
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
F=tests/fixture

echo "=== [0/8] fixture ==="
"$PY" tests/make_fixture.py --out "$F"

echo "=== [1/8] linear (run_full_gadi.py) ==="
"$PY" run_full_gadi.py --parquet "$F/spectra.parquet" --allstar "$F/allStar.fits" \
    --pixel-mask-dir "$F/pixel_masks" --out "$F/out_linear.parquet" --lam 0.1
"$PY" tests/checks.py linear "$F"

echo "=== [2/8] het-NN (train_nn.py, 2 epochs) ==="
"$PY" train_nn.py --parquet "$F/spectra.parquet" --allstar "$F/allStar.fits" \
    --pixel-mask-dir "$F/pixel_masks" --out "$F/out_nn.parquet" \
    --epochs 2 --hidden 32 --batch-size 64 --device cpu
"$PY" tests/checks.py nn "$F"

echo "=== [3/8] v2 stage-1 (train_v2.py) ==="
# --batch-size 32 is mandatory: v2 drops the ragged tail batch, so a batch size
# larger than the tiny fixture fold would train on ZERO batches and "pass".
"$PY" train_v2.py --parquet "$F/spectra.parquet" --allstar "$F/allStar.fits" \
    --pixel-mask-dir "$F/pixel_masks" --out "$F/out_v2.parquet" \
    --stages 1 --epochs1 2 --batch-size 32 --pca-k 8 \
    --g-hidden 8 --s-hidden 8 --device cpu
"$PY" tests/checks.py v2 "$F"

echo "=== [4/8] apply NN to the cluster (apply_nn.py) ==="
"$PY" apply_nn.py --model "$F/out_nn_model.pt" --parquet "$F/cluster_spectra.parquet" \
    --allstar "$F/allStar.fits" --pixel-mask-dir "$F/pixel_masks" \
    --cluster-name FAKE1 --out "$F/fake1_nn.parquet" --device cpu
"$PY" tests/checks.py applied "$F"

echo "=== [5/8] cluster test (run_cluster_test.py) ==="
"$PY" run_cluster_test.py "$F/fake1_nn.parquet" --min-members 5
"$PY" tests/checks.py cluster "$F"

echo "=== [6/8] old-checkpoint back-compat (apply with stripped keys) ==="
"$PY" tests/checks.py strip-dataset "$F/out_nn_model.pt" "$F/old_style.pt"
"$PY" apply_nn.py --model "$F/old_style.pt" --parquet "$F/cluster_spectra.parquet" \
    --allstar "$F/allStar.fits" --pixel-mask-dir "$F/pixel_masks" \
    --cluster-name FAKE1 --out "$F/fake1_nn_old.parquet" --device cpu
"$PY" tests/checks.py compare-plx "$F/fake1_nn.parquet" "$F/fake1_nn_old.parquet"

echo "=== [7/8] cluster spectra prep (prepare_cluster_spectra.py) ==="
"$PY" prepare_cluster_spectra.py --cluster-dir "$F/cluster_meta" \
    --spectra "$F/cluster_spectra.parquet" --out-dir "$F/spectra_per_cluster"
"$PY" tests/checks.py prep "$F"

echo "=== [8/8] value digests (compare across commits) ==="
"$PY" tests/checks.py digest "$F"

echo "SMOKE TEST PASSED"
