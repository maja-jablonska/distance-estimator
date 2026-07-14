#!/usr/bin/env python3
"""
checks.py — assertion helpers called between tests/smoke_test.sh steps.

    python tests/checks.py linear  <fixture_dir>     # run_full_gadi outputs
    python tests/checks.py nn      <fixture_dir>     # train_nn outputs
    python tests/checks.py v2      <fixture_dir>     # train_v2 outputs
    python tests/checks.py applied <fixture_dir>     # apply_nn output
    python tests/checks.py cluster <fixture_dir>     # cluster-test metrics
    python tests/checks.py prep    <fixture_dir>     # prepare_cluster_spectra output
    python tests/checks.py digest  <fixture_dir>     # value digests (cross-commit oracle)
    python tests/checks.py strip-dataset <in.pt> <out.pt>   # simulate an OLD checkpoint
    python tests/checks.py compare-plx <a.parquet> <b.parquet>

Each subcommand raises (non-zero exit) on failure, so smoke_test.sh's `set -e`
stops at the first broken invariant.
"""
from __future__ import annotations
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# must match tests/make_fixture.py's design (see its docstring)
EXPECTED_GOOD = 274          # 300 - 20 (chip gap) - 3 - 3 (telescope dead pixels)
N_CLUSTER = 30


def _read(path):
    import pandas as pd
    assert os.path.exists(path), f"missing output: {path}"
    df = pd.read_parquet(path)
    assert len(df) > 0, f"{path} is empty"
    return df


def _check_plx_sp(df, path):
    p = df["plx_sp"].to_numpy()
    assert np.isfinite(p).all(), f"{path}: non-finite plx_sp"
    assert (p > 0).all(), f"{path}: non-positive plx_sp"
    e = df["err_sp"].to_numpy()
    assert np.isfinite(e).all() and (e > 0).all(), f"{path}: bad err_sp"


def check_linear(fix):
    import run_full_gadi as R
    path = os.path.join(fix, "out_linear.parquet")
    df = _read(path)
    _check_plx_sp(df, path)
    assert set(df["sample"]) == {"A", "B"}, "missing fold in sample column"
    assert df["train"].any() and (~df["train"]).any(), "train mask degenerate"
    m = R.load_model(os.path.join(fix, "out_linear_model.npz"))
    assert m["label_cols"] == ["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag",
                               "k_mag", "w1_mag", "w2_mag"], m["label_cols"]
    assert int(m["good_pixel_mask"].sum()) == EXPECTED_GOOD, \
        f"good pixels {int(m['good_pixel_mask'].sum())} != {EXPECTED_GOOD}"
    assert m["tel_masks"] is not None and set(m["tel_masks"]) == {"apo25m", "lco25m"}
    assert m["dataset"].name == "dr17", m["dataset"]
    print(f"linear OK: {len(df)} stars, good={EXPECTED_GOOD}, "
          f"scatter(plx_sp/plx-1) exercised")


def check_nn(fix):
    import torch
    path = os.path.join(fix, "out_nn.parquet")
    df = _read(path)
    _check_plx_sp(df, path)
    ckpt = torch.load(os.path.join(fix, "out_nn_model.pt"),
                      map_location="cpu", weights_only=False)
    for k in ("state_dict", "mu", "sd", "good_pixel_mask", "label_cols", "d_in",
              "hidden", "std_factor", "dataset", "phot_cols"):
        assert k in ckpt, f"nn checkpoint missing key {k}"
    assert int(ckpt["d_in"]) == 8 + EXPECTED_GOOD, ckpt["d_in"]
    assert ckpt["dataset"] == "dr17", ckpt["dataset"]
    print(f"nn OK: {len(df)} stars, d_in={ckpt['d_in']}, dataset={ckpt['dataset']}")


def check_v2(fix):
    path = os.path.join(fix, "out_v2.parquet")
    df = _read(path)
    _check_plx_sp(df, path)
    for col in ("a_ks_rjce", "hull_d", "gate"):
        assert col in df.columns, f"v2 parquet missing {col}"
    z = np.load(os.path.join(fix, "out_v2_model.npz"), allow_pickle=False)
    for k in ("theta", "b", "phi", "pca_V", "mu", "sd", "good_pixel_mask",
              "label_cols", "dataset", "phot_cols", "n_phot", "has_aks"):
        assert k in z.files, f"v2 checkpoint missing {k}"
    assert z["theta"].shape[0] == 8 + 1 + EXPECTED_GOOD, z["theta"].shape
    assert str(z["dataset"]) == "dr17" and int(z["n_phot"]) == 8 and bool(z["has_aks"])
    print(f"v2 OK: {len(df)} stars, theta dim={z['theta'].shape[0]}, "
          f"dataset={z['dataset']}")


def check_applied(fix, name="fake1_nn.parquet"):
    path = os.path.join(fix, name)
    df = _read(path)
    _check_plx_sp(df, path)
    assert len(df) == N_CLUSTER, f"{path}: {len(df)} rows != {N_CLUSTER}"
    assert "cluster" in df.columns and (df["cluster"] == "FAKE1").all()
    assert "memberprob" in df.columns and df["memberprob"].between(0, 1).all()
    print(f"applied OK: {len(df)} cluster stars, memberprob carried through")


def check_cluster(fix):
    import cluster_test as C
    df = _read(os.path.join(fix, "fake1_nn.parquet"))
    members = C.members_from_labels(df, "cluster")
    rep = C.run_cluster_test(df, members, min_members=5)
    assert "FAKE1" in rep, f"cluster test skipped FAKE1: {list(rep)}"
    r = rep["FAKE1"]
    assert np.isfinite(r["internal_chi2"]), r
    assert np.isfinite(r["sp_mean"]) and r["sp_mean"] > 0, r
    print(f"cluster OK: N={r['n']} sp_mean={r['sp_mean']:.3f} "
          f"intchi2={r['internal_chi2']:.2f}")


def check_prep(fix):
    df = _read(os.path.join(fix, "spectra_per_cluster", "FAKE1.parquet"))
    for c in ("sdss_id", "flux", "continuum", "ivar", "telescope"):
        assert c in df.columns, f"prepared cluster parquet missing {c}"
    print(f"prep OK: {len(df)} member spectra for FAKE1")


def digest(fix):
    """Value digests of the key outputs — the cross-commit regression oracle.
    (Hashes rounded values, not file bytes: parquet/npz containers embed
    library-version metadata that would change the byte hash spuriously.)"""
    import pandas as pd

    def h(a):
        return hashlib.sha256(np.round(np.asarray(a, np.float64), 8).tobytes()
                              ).hexdigest()[:16]

    for name in ("out_linear.parquet", "out_nn.parquet", "out_v2.parquet",
                 "fake1_nn.parquet"):
        p = os.path.join(fix, name)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            print(f"  {name:<24} plx_sp {h(df['plx_sp'])}  err_sp {h(df['err_sp'])}")
    z = np.load(os.path.join(fix, "out_linear_model.npz"), allow_pickle=False)
    print(f"  {'out_linear_model.npz':<24} theta_all {h(z['theta_all'])}")


def strip_dataset(src, dst):
    """Simulate a pre-refactor checkpoint: drop the dataset/phot_cols keys."""
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    removed = [k for k in ("dataset", "phot_cols") if ckpt.pop(k, None) is not None]
    torch.save(ckpt, dst)
    print(f"stripped {removed or 'nothing'} -> {dst}")


def compare_plx(a, b):
    import pandas as pd
    pa_, pb = pd.read_parquet(a), pd.read_parquet(b)
    assert np.allclose(pa_["plx_sp"], pb["plx_sp"], rtol=1e-10), \
        f"plx_sp differs between {a} and {b}"
    assert np.allclose(pa_["err_sp"], pb["err_sp"], rtol=1e-10), \
        f"err_sp differs between {a} and {b}"
    print(f"compare OK: {a} == {b}")


CMDS = {"linear": check_linear, "nn": check_nn, "v2": check_v2,
        "applied": check_applied, "cluster": check_cluster, "prep": check_prep,
        "digest": digest, "strip-dataset": strip_dataset, "compare-plx": compare_plx}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    CMDS[sys.argv[1]](*sys.argv[2:])
