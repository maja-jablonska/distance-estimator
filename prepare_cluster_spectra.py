#!/usr/bin/env python3
"""
prepare_cluster_spectra.py — turn the per-cluster *metadata* parquets from
cluster.ipynb into the per-cluster *spectra* parquets the distance-estimation
harness (apply_nn.pbs -> apply_nn.py) consumes.

The gap this bridges
--------------------
cluster.ipynb crossmatches each cluster's member list against
`astra_photogeo.parquet` and writes `<cluster>_astra_matched.parquet`. Those are
astra METADATA rows (keyed by gaia_dr3_source_id, carrying sdss_id + the
Bailer-Jones photogeo distance) — they do NOT hold the spectra. But
apply_nn.py builds its ln-flux features by streaming a spectra parquet with the
array columns flux/continuum/ivar (+ telescope/snr/spectrum_flags/zeropoint), the
same schema as the training file spectra_infer_parallax_zpt.parquet.

So for each cluster we must pull the member spectra out of the big spectra
parquet, apply the data-quality cuts, keep ONE clean spectrum per star, and write
`<out_dir>/<cluster>.parquet`. That directory is exactly what apply_nn.pbs loops
over (CLUSTER_DIR), so after this the existing harness runs unchanged:

    qsub -v CLUSTER_DIR=<out_dir>,MODEL=<...>_model.pt apply_nn.pbs

What "all the cuts" means here
------------------------------
The distance estimation applies cuts at several stages; this prep owns only the
ones that select WHICH rows/spectra reach the model, and leaves the rest to the
existing, model-matched code:

  * one clean spectrum per star  (here)   — collapse multi-visit rows to the
        highest-snr spectrum, preferring spectrum_flags == 0, mirroring
        load_metadata's per-sdss_id dedup.
  * spectrum_flags == 0          (here, default on) — APOGEE data-quality flag,
        the same clean-spectrum cut the training set used (run_full_gadi).
  * optional snr floor           (here, default OFF) — the training snr>100 cut is
        a TRAINING-set cut, not an apply-time cut; for the cluster test we want
        faint members, so this is off unless you ask for it.
  * complete photometry (8 mags) (downstream) — apply_nn.py applies this itself
        from the allStar table; the mags aren't in the spectra parquet, so we do
        NOT duplicate it here (we only optionally report it if --allstar given).
  * per-pixel good-pixel mask / telescope lit masks / f_max / bad-frac
        (downstream) — build_lnflux_streaming applies the model's SAVED mask so
        new spectra land on exactly the pixels the model was fit on. Not a prep
        concern.

Membership keying
-----------------
The big spectra parquet is keyed by sdss_id (see data.ipynb), and astra_photogeo
(hence every *_astra_matched.parquet) carries sdss_id too, so we join on sdss_id.
Pass --id-col to override.

Usage
-----
    python prepare_cluster_spectra.py \
        --cluster-dir  /scratch/.../clusters/matched_astra_parquets \
        --spectra      /scratch/.../data/spectra_infer_parallax_zpt.parquet \
        --out-dir      /scratch/.../clusters/spectra_per_cluster

Then point apply_nn.pbs at --out-dir:
    qsub -v CLUSTER_DIR=/scratch/.../clusters/spectra_per_cluster,MODEL=... apply_nn.pbs
"""
from __future__ import annotations
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Columns apply_nn.py needs downstream:
#   load_metadata        -> sdss_id, snr, spectrum_flags, (zeropoint)
#   build_lnflux_streaming -> flux, continuum, ivar, telescope
SPEC_COLS_REQUIRED = ["sdss_id", "flux", "continuum", "ivar", "telescope"]
SPEC_COLS_OPTIONAL = ["snr", "spectrum_flags", "zeropoint"]


# ----------------------------------------------------------------------
# stage 1 — read each cluster's member ids from the astra-matched parquet
# ----------------------------------------------------------------------
def _read_catalogue_probs(cat_path):
    """{Gaia source_id: membership probability} from a Vasiliev & Baumgardt (2021)
    globular-cluster catalogue .txt (see clusters/!readme.txt): column 0 is the
    Gaia (E)DR3 source_id, the LAST column is the membership probability. The header
    line is commented with '#'. Column 0 is forced to int64 so the 19-digit source
    ids are not silently truncated through float."""
    import pandas as pd
    df = pd.read_csv(cat_path, sep=r"\s+", comment="#", header=None,
                     dtype={0: "int64"}, low_memory=False)
    sid = df.iloc[:, 0].to_numpy()
    prob = df.iloc[:, -1].astype(float).to_numpy()
    return dict(zip(sid.tolist(), prob.tolist()))


def load_cluster_members(cluster_dir, id_col="sdss_id", gaia_id_col="gaia_dr3_source_id",
                         catalogue_dir=None, prob_min=0.0,
                         pattern="*_astra_matched.parquet", strip_suffix="_astra_matched"):
    """Return ({cluster_name: int64 member id array}, {member id: memberprob}).

    Cluster name is the file stem with `strip_suffix` removed (so
    `NGC_3201_astra_matched.parquet` -> `NGC_3201`), which becomes the output
    parquet name apply_nn.pbs labels members with.

    Membership probability: cluster.ipynb matched the catalogues on source_id with
    NO probability cut, so the matched parquets still contain memberprob~0 field
    stars. When `catalogue_dir` is given, each cluster's <name>.txt is re-joined by
    `gaia_id_col` (gaia_dr3_source_id; the catalogues are Gaia EDR3 == DR3 keyed)
    and only members with memberprob >= `prob_min` are kept. The probabilities are
    returned too, so prepare() can carry a `memberprob` column through to the
    inferred catalogs for inspection / probability weighting in the test.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(cluster_dir, pattern)))
    if not files:
        # fall back to any parquet in the dir, so a custom layout still works
        files = sorted(glob.glob(os.path.join(cluster_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files found in {cluster_dir}")

    use_prob = catalogue_dir is not None
    members, prob_lut = {}, {}
    n_no_cat = 0
    for f in files:
        name = Path(f).stem
        if strip_suffix and name.endswith(strip_suffix):
            name = name[: -len(strip_suffix)]
        avail = set(pq.ParquetFile(f).schema_arrow.names)
        if id_col not in avail:
            raise SystemExit(
                f"{f} has no '{id_col}' column (available: {sorted(avail)[:12]}...). "
                f"Pass --id-col with the join key shared by these parquets and the "
                f"spectra parquet.")
        read_cols = [id_col]
        if use_prob:
            if gaia_id_col not in avail:
                raise SystemExit(
                    f"{f} has no '{gaia_id_col}' column, needed to join membership "
                    f"probability. Pass --gaia-id-col (available: {sorted(avail)[:12]}...).")
            read_cols.append(gaia_id_col)
        t = pq.read_table(f, columns=read_cols).to_pandas()
        sdss = np.asarray(t[id_col].to_numpy())
        sdss = sdss[np.isfinite(sdss)] if np.issubdtype(sdss.dtype, np.floating) else sdss
        sdss = sdss.astype(np.int64)

        if use_prob:
            cat_path = os.path.join(catalogue_dir, f"{name}.txt")
            gid = np.asarray(t[gaia_id_col].to_numpy())
            if not os.path.exists(cat_path):
                log(f"  WARNING {name}: catalogue {cat_path} not found "
                    f"-> probability cut NOT applied for this cluster")
                n_no_cat += 1
                probs = np.full(len(sdss), np.nan)
            else:
                lut = _read_catalogue_probs(cat_path)
                # gaia ids may be int64, nullable Int64, or float with NaN
                def gkey(g):
                    if g is None or (isinstance(g, float) and not np.isfinite(g)):
                        return None
                    return int(g)
                probs = np.array([lut.get(gkey(g), np.nan) for g in gid], float)
            keep = probs >= prob_min                      # NaN >= x is False -> dropped
            n_before = len(sdss)
            sdss, probs = sdss[keep], probs[keep]
            for s, p in zip(sdss.tolist(), probs.tolist()):
                prob_lut[int(s)] = float(p)
            log(f"  {name:<24} {len(sdss):>6} / {n_before:<6} members "
                f"(memberprob >= {prob_min:g})")
        else:
            log(f"  {name:<24} {len(np.unique(sdss)):>6} members (no probability cut)")

        members[name] = np.unique(sdss)

    if use_prob and n_no_cat:
        log(f"NOTE: {n_no_cat} cluster(s) had no catalogue file -> kept unfiltered")

    # warn on members claimed by >1 cluster (rare for GCs; they'd be written twice)
    seen, dup = set(), set()
    for ids in members.values():
        for i in ids.tolist():
            (dup if i in seen else seen).add(i)
    if dup:
        log(f"NOTE: {len(dup)} member id(s) appear in >1 cluster -> written to each")
    log(f"loaded {len(members)} clusters, {len(seen)} unique member ids total")
    return members, prob_lut


# ----------------------------------------------------------------------
# stage 2 — stream the big spectra parquet, keep only member rows
# ----------------------------------------------------------------------
def fetch_member_spectra(spectra_path, member_ids, id_col="sdss_id", batch_rows=20000):
    """One streaming pass over the spectra parquet, returning a pyarrow Table of
    only the rows whose `id_col` is in `member_ids` (the union across all clusters).

    Streams in batches so the multi-GB / 800k-row spectra file is never held in
    memory; only the matched rows (a few thousand) are accumulated.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(spectra_path)
    avail = set(pf.schema_arrow.names)
    if id_col not in avail:
        raise SystemExit(f"spectra parquet has no '{id_col}' column "
                         f"(available: {sorted(avail)[:12]}...)")
    missing_req = [c for c in SPEC_COLS_REQUIRED if c not in avail]
    if missing_req:
        raise SystemExit(f"spectra parquet missing required columns {missing_req}; "
                         f"available: {sorted(avail)}")
    cols = list(SPEC_COLS_REQUIRED) + [c for c in SPEC_COLS_OPTIONAL if c in avail]
    no_zpt = "zeropoint" not in avail
    if no_zpt:
        log("spectra parquet has no 'zeropoint' -> apply_nn will use plx_raw; pass "
            "OFFSET to the cluster test (see CLUSTER_TEST.md gotchas)")

    want = pa.array(np.asarray(member_ids, np.int64))
    parts = []
    n_scanned = n_hit = 0
    t0 = time.time()
    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows, columns=cols)):
        n_scanned += batch.num_rows
        mask = pc.is_in(batch.column(id_col), value_set=want)
        k = pc.sum(mask).as_py() or 0
        if k:
            parts.append(pa.Table.from_batches([batch]).filter(mask))
            n_hit += k
        if bi % 50 == 0:
            log(f"  scanned {n_scanned} rows, matched {n_hit} "
                f"({time.time()-t0:.0f}s)")
    if not parts:
        raise SystemExit("no member spectra matched — check --id-col / spectra path")
    tbl = pa.concat_tables(parts)
    log(f"matched {tbl.num_rows} spectra rows for "
        f"{len(set(tbl.column(id_col).to_pylist()))} unique stars "
        f"(scanned {n_scanned})")
    return tbl, no_zpt


# ----------------------------------------------------------------------
# stage 3 — cuts + one-clean-spectrum-per-star + per-cluster write
# ----------------------------------------------------------------------
def prepare(cluster_dir, spectra_path, out_dir, *, id_col="sdss_id",
            gaia_id_col="gaia_dr3_source_id", catalogue_dir=None, prob_min=0.0,
            require_clean=True, snr_min=0.0, batch_rows=20000):
    """End-to-end: members -> member spectra -> cuts/dedup -> per-cluster parquets."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("stage 1: reading cluster member ids ...")
    members, prob_lut = load_cluster_members(
        cluster_dir, id_col=id_col, gaia_id_col=gaia_id_col,
        catalogue_dir=catalogue_dir, prob_min=prob_min)
    if not any(len(v) for v in members.values()):
        raise SystemExit("no members survive the membership-probability cut "
                         f"(prob_min={prob_min}); lower --prob-min or check --catalogue-dir")
    all_ids = np.unique(np.concatenate(list(members.values())))

    log(f"stage 2: streaming spectra parquet for {len(all_ids)} member ids ...")
    tbl, no_zpt = fetch_member_spectra(spectra_path, all_ids, id_col=id_col,
                                       batch_rows=batch_rows)
    spec = tbl.to_pandas()

    # --- cut 1: clean spectra (spectrum_flags == 0), the training data-quality cut ---
    if require_clean and "spectrum_flags" in spec.columns:
        before = len(spec)
        spec = spec[spec["spectrum_flags"].to_numpy() == 0]
        log(f"cut spectrum_flags==0: {len(spec)}/{before} rows kept")
    elif require_clean:
        log("WARNING: no 'spectrum_flags' column -> clean-spectrum cut skipped")

    # --- cut 2: optional snr floor (OFF by default; training-only cut) ---
    if snr_min > 0 and "snr" in spec.columns:
        before = len(spec)
        spec = spec[spec["snr"].to_numpy() >= snr_min]
        log(f"cut snr>={snr_min:g}: {len(spec)}/{before} rows kept")
    elif snr_min > 0:
        log("WARNING: no 'snr' column -> snr cut skipped")

    # --- collapse to ONE spectrum per star: highest snr wins (clean already enforced) ---
    n_rows = len(spec)
    if "snr" in spec.columns:
        spec = (spec.sort_values("snr", ascending=False, kind="stable")
                    .drop_duplicates(id_col, keep="first"))
    else:
        spec = spec.drop_duplicates(id_col, keep="first")
    n_dup = n_rows - len(spec)
    if n_dup:
        log(f"collapsed {n_dup} extra visit/reduction rows -> one spectrum per star")
    # carry membership probability through so the inferred catalogs keep it
    if prob_lut:
        spec["memberprob"] = [prob_lut.get(int(i), np.nan)
                              for i in spec[id_col].to_numpy()]
    spec_by_id = {int(i): k for k, i in enumerate(spec[id_col].to_numpy())}
    log(f"{len(spec)} unique stars with a clean spectrum ready")

    # --- stage 4: write one parquet per cluster (filename == cluster name) ---
    log(f"stage 3: writing per-cluster spectra parquets -> {out_dir}")
    summary = []
    for name, ids in members.items():
        pos = [spec_by_id[i] for i in ids.tolist() if i in spec_by_id]
        sub = spec.iloc[pos]
        summary.append((name, len(ids), len(sub)))
        out_path = out_dir / f"{name}.parquet"
        if len(sub) == 0:
            # don't write an empty parquet: build_lnflux_streaming has no batch to
            # infer the spectral grid from and trips the saved-mask assertion. Also
            # clear a stale empty file from a previous run so --skip-prep stays clean.
            if out_path.exists():
                out_path.unlink()
            log(f"  {name:<24}     0/{len(ids):<5} members with spectra -> SKIP (no spectra)")
            continue
        sub.to_parquet(out_path, index=False)
        log(f"  {name:<24} {len(sub):>5}/{len(ids):<5} members with spectra "
            f"-> {out_path.name}")

    log("=== summary (cluster: members_with_spectra / catalog_members) ===")
    tot_m = tot_s = 0
    for name, n_mem, n_spec in sorted(summary, key=lambda r: -r[2]):
        frac = 100 * n_spec / n_mem if n_mem else 0
        log(f"  {name:<24} {n_spec:>5} / {n_mem:<5}  ({frac:4.0f}%)")
        tot_m += n_mem
        tot_s += n_spec
    log(f"  {'TOTAL':<24} {tot_s:>5} / {tot_m:<5}")
    if no_zpt:
        log("reminder: no zeropoint -> run the cluster test with PLX_G=plx_raw and the "
            "global OFFSET (see CLUSTER_TEST.md)")
    log(f"done. feed {out_dir} to apply_nn.pbs as CLUSTER_DIR.")
    return summary


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-dir", required=True,
                    help="dir of <cluster>_astra_matched.parquet (cluster.ipynb output)")
    ap.add_argument("--spectra", required=True,
                    help="big spectra parquet with flux/continuum/ivar (e.g. "
                         "spectra_infer_parallax_zpt.parquet)")
    ap.add_argument("--out-dir", required=True,
                    help="where per-cluster spectra parquets are written "
                         "(feed this to apply_nn.pbs as CLUSTER_DIR)")
    ap.add_argument("--id-col", default="sdss_id",
                    help="join key shared by the cluster parquets and the spectra "
                         "parquet (default sdss_id)")
    ap.add_argument("--catalogue-dir", default=None,
                    help="dir of Vasiliev&Baumgardt <cluster>.txt catalogues; enables "
                         "the membership-probability cut (memberprob = last column)")
    ap.add_argument("--prob-min", type=float, default=0.9,
                    help="membership probability threshold (default 0.9; only applied "
                         "when --catalogue-dir is given)")
    ap.add_argument("--gaia-id-col", default="gaia_dr3_source_id",
                    help="gaia source-id column in the cluster parquets used to join "
                         "the catalogue membership probability")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="do NOT cut on spectrum_flags==0 (default keeps only clean)")
    ap.add_argument("--snr-min", type=float, default=0.0,
                    help="optional snr floor (default 0 = off; training used 100, but "
                         "that is a training-only cut)")
    ap.add_argument("--batch-rows", type=int, default=20000,
                    help="streaming batch size over the spectra parquet")
    args = ap.parse_args()

    prepare(args.cluster_dir, args.spectra, args.out_dir,
            id_col=args.id_col, gaia_id_col=args.gaia_id_col,
            catalogue_dir=args.catalogue_dir, prob_min=args.prob_min,
            require_clean=not args.keep_flagged,
            snr_min=args.snr_min, batch_rows=args.batch_rows)


if __name__ == "__main__":
    main()
