#!/usr/bin/env python3
"""
hogg18_cluster_test.py — repeat Hogg+18's Figure-4 cluster validation on the
original DR14 dataset.

Paper I (sec. 5, Fig. 4) validated the spectrophotometric parallaxes against
three clusters spanning metallicity/age/extinction — "NGC2862" (their label;
the ~1.1 mas line identifies it as M67/NGC 2682), M71 and M107 — with members
"found by hand, looking at sky and proper-motion distributions centered on
literature values". This script repeats that selection mechanically:

  member := parent-sample star  AND  sky separation < radius
            AND |pm - pm_literature| < pm_tol   (Gaia DR2 proper motions)

No parallax is used in the selection (the paper's commitment), so the test is
independent of the quantity being validated. Proper motions are not in
hogg18_meta.parquet; they are fetched once from the ESA archive by source_id
(numeric IN-lists < 2000 per sync query — the route that works, see
assemble_hogg18.py) and cached next to the meta file.

Predictions scored:
  * paper : the Zenodo spec_parallax / spec_parallax_err columns (always
            available in the meta parquet) — reproduces Fig. 4 itself.
  * ours  : plx_sp / err_sp from a run_full_gadi.py results parquet
            (--results), matched on sdss_id.

The Gaia side is the RAW DR2 parallax, exactly as plotted in Fig. 4; the
paper's +0.048 mas zero-point (which both spec_parallax and our plx_sp are
trained to include) enters the metrics via cluster_metrics(offset=+0.048).

Usage:
  python hogg18_cluster_test.py                          # paper predictions
  python hogg18_cluster_test.py --results spphot_results_hogg18.parquet
  # (copy the results from Gadi first:
  #  scp gadi:/scratch/mk27/mj8805/hogg18/spphot_results_hogg18.parquet .)
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np

from spphot.clusters import cluster_metrics
from spphot.data import log

ZEROPOINT_MAS = 0.048           # paper's adopted Gaia DR2 parallax offset

# literature centers + Gaia DR2-era mean proper motions (SIMBAD; PMs from
# Gaia DR2 / Vasiliev & Baumgardt). radius/pm_tol chosen so the selection is
# conservative ("only very securely identified members", like the paper):
# field stars at these |b| have PM dispersion ~5 mas/yr, so a 1 mas/yr circle
# around the cluster PM is a strong cut; radii cover the APOGEE fibers that
# targeted each cluster (M67 is nearby -> members spread over ~1 degree).
CLUSTERS = {
    #            ra [deg]  dec [deg]  r [deg]  pmra    pmdec  [mas/yr]  tol
    "M67":  dict(ra=132.846, dec=+11.814, radius=1.00, pmra=-10.97, pmdec=-2.94, pm_tol=1.0),
    "M71":  dict(ra=298.444, dec=+18.779, radius=0.50, pmra=-3.41,  pmdec=-2.61, pm_tol=1.0),
    "M107": dict(ra=248.133, dec=-13.054, radius=0.50, pmra=-1.93,  pmdec=-5.98, pm_tol=1.0),
}


def fetch_proper_motions(source_ids, cache_path):
    """{source_id: (pmra, pmdec)} from gaiadr2.gaia_source, cached to parquet."""
    import pandas as pd
    source_ids = np.unique(np.asarray(source_ids, np.int64))
    if os.path.exists(cache_path):
        pm = pd.read_parquet(cache_path)
        missing = np.setdiff1d(source_ids, pm["source_id"].to_numpy())
        if missing.size == 0:
            log(f"proper motions: all {len(source_ids)} candidates cached")
            return pm
    else:
        pm, missing = None, source_ids

    from astroquery.gaia import Gaia
    parts = [] if pm is None else [pm]
    step = 1900                                   # sync-query IN-list limit
    for i in range(0, len(missing), step):
        inlist = ",".join(str(s) for s in missing[i:i + step])
        for attempt in range(5):
            try:
                job = Gaia.launch_job(
                    "SELECT source_id, pmra, pmdec, pmra_error, pmdec_error "
                    f"FROM gaiadr2.gaia_source WHERE source_id IN ({inlist})")
                chunk = job.get_results().to_pandas()
                chunk.columns = [c.lower() for c in chunk.columns]
                parts.append(chunk)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 10 * 2 ** attempt
                log(f"  Gaia PM chunk: {type(e).__name__}, retry in {wait}s ...")
                time.sleep(wait)
        log(f"  Gaia PM: {sum(len(p) for p in parts)}/{len(source_ids)} fetched")
    pm = pd.concat(parts, ignore_index=True).drop_duplicates("source_id")
    pm.to_parquet(cache_path, index=False)
    return pm


def select_members(meta, pm):
    """{cluster: integer row positions into meta} via sky cone + PM comovement."""
    pmmap = pm.set_index("source_id")
    ra, dec = meta["ra"].to_numpy(float), meta["dec"].to_numpy(float)
    sid = meta["source_id"].to_numpy()
    members = {}
    for name, c in CLUSTERS.items():
        sep = np.hypot((ra - c["ra"]) * np.cos(np.radians(c["dec"])), dec - c["dec"])
        cand = np.where(sep < c["radius"])[0]
        keep = []
        for i in cand:
            try:
                row = pmmap.loc[int(sid[i])]
            except (KeyError, ValueError):
                continue                          # no Gaia PM -> not securely a member
            dpm = np.hypot(row["pmra"] - c["pmra"], row["pmdec"] - c["pmdec"])
            if np.isfinite(dpm) and dpm < c["pm_tol"]:
                keep.append(i)
        members[name] = np.array(keep, int)
        log(f"{name}: {len(cand)} in {c['radius']:g} deg cone -> "
            f"{len(keep)} comoving within {c['pm_tol']:g} mas/yr "
            f"(paper Fig. 4 plotted {dict(M67=5, M71=11, M107=10)[name]})")
    return members


def report(meta, members, plx_sp, err_sp, tag, logg=None, detail=True):
    """Per-cluster metrics table for one set of predictions, plus (detail=True)
    a per-member breakdown sorted by |chi| so the internal-chi2 drivers are
    visible: chi = (member sp - cluster IVW sp mean) / member err_sp."""
    print(f"\n=== cluster test [{tag}]  (Gaia = raw DR2; offset=+{ZEROPOINT_MAS} "
          f"applied to the comparison, as trained) ===")
    print(f"  {'cluster':<6} {'N':>3} {'sp_mean':>8} {'gaia_raw':>9} "
          f"{'Δ(sp-gaia-zp)':>13} {'σ':>6} {'tight%':>7} {'intχ²':>6}")
    out = {}
    snr = meta["snr"].to_numpy(float)
    plx_a = meta["plx"].to_numpy(float)
    err_a = meta["e_plx"].to_numpy(float)
    ids = meta["sdss_id"].to_numpy()
    for name, idx in members.items():
        idx = idx[np.isfinite(plx_sp[idx])]
        r = cluster_metrics(plx_sp[idx], err_sp[idx], plx_a[idx], err_a[idx],
                            offset=+ZEROPOINT_MAS)   # sp is trained on plx_raw+zp
        out[name] = r
        print(f"  {name:<6} {r['n']:>3} {r['sp_mean']:>8.3f} {r['gaia_mean']:>9.3f} "
              f"{1e3 * (r['sp_mean'] - r['gaia_mean'] - ZEROPOINT_MAS):>+9.0f} uas "
              f"{r['sp_vs_gaia_sigma']:>+5.1f} {100 * r['tightness_frac']:>6.1f}% "
              f"{r['internal_chi2']:>6.1f}")
    print("  σ: spec-vs-Gaia IVW-mean offset in combined sigma (|σ|<~2-3 ok)")
    print("  intχ²: member spread / quoted err (Gaia-free; >>1 = overconfident errors)")
    if detail:
        for name, idx in members.items():
            idx = idx[np.isfinite(plx_sp[idx])]
            chi = (plx_sp[idx] - out[name]["sp_mean"]) / err_sp[idx]
            order = np.argsort(-np.abs(chi))
            print(f"\n  -- {name} members by |chi| ({tag}); "
                  f"chi2 contributions sum to N*intχ² --")
            print(f"     {'sdss_id':<20} {'S/N':>5} {'logg':>5} "
                  f"{'gaia±err':>15} {'sp±err':>15} {'chi':>6}")
            for j in order:
                i = idx[j]
                lg = logg[i] if logg is not None else meta['logg'].to_numpy(float)[i]
                print(f"     {str(ids[i]):<20} {snr[i]:>5.0f} {lg:>5.2f} "
                      f"{plx_a[i]:>7.3f}±{err_a[i]:<6.3f} "
                      f"{plx_sp[i]:>7.3f}±{err_sp[i]:<6.3f} {chi[j]:>+6.1f}")
    return out


def member_logg(meta, allstar_raw):
    """Calibrated logg, falling back to uncalibrated FPARAM logg where DR14
    ASPCAP left the -9999 sentinel (metal-poor globular giants — 7 of the 11
    M71 members; the paper's Fig. 4 y-values can only be the FPARAM ones)."""
    logg = meta["logg"].to_numpy(float).copy()
    bad = ~(logg > 0)
    if bad.any() and allstar_raw and os.path.exists(allstar_raw):
        from astropy.io import fits
        a = fits.open(allstar_raw, memmap=True)[1].data
        ids = np.char.strip(a["APOGEE_ID"].astype(str))
        lut = {i: k for k, i in enumerate(ids)}
        fp = a["FPARAM"]
        for j in np.where(bad)[0]:
            k = lut.get(meta["sdss_id"].iloc[j])
            if k is not None and np.isfinite(fp[k, 1]) and fp[k, 1] > -100:
                logg[j] = fp[k, 1]
    logg[~(logg > 0)] = np.nan
    return logg


def fig4(meta, members, plx_sp, err_sp, logg, tag, path):
    """The Fig-4 plot: per member, spec (black) and raw-Gaia (grey) parallax vs
    log g, with dashed vertical IVW means, three stacked panels sharing x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spphot.clusters import ivw_mean

    plx_a = meta["plx"].to_numpy(float)
    err_a = meta["e_plx"].to_numpy(float)
    fig, axes = plt.subplots(len(members), 1, figsize=(7, 2.2 * len(members)),
                             sharex=True, gridspec_kw={"hspace": 0.05})
    for ax, (name, idx) in zip(np.atleast_1d(axes), members.items()):
        idx = idx[np.isfinite(plx_sp[idx])]
        ax.errorbar(plx_sp[idx], logg[idx], xerr=err_sp[idx], fmt="o", ms=4,
                    color="k", lw=1, label=r"spectrophotometric $\varpi^{(sp)}$")
        ax.errorbar(plx_a[idx], logg[idx], xerr=err_a[idx], fmt="o", ms=4,
                    color="0.6", lw=1, label=r"Gaia DR2 $\varpi^{(a)}$")
        for vals, errs, c in ((plx_sp[idx], err_sp[idx], "k"),
                              (plx_a[idx], err_a[idx], "0.6")):
            mean, _ = ivw_mean(vals, errs)
            ax.axvline(mean, color=c, ls=":", lw=1)
        ax.text(0.98, 0.1, name, transform=ax.transAxes, ha="right")
        ax.set_ylabel(r"$\log g$")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel(r"$\varpi$ [mas]")
    np.atleast_1d(axes)[0].set_title(f"Hogg+18 Fig. 4 repetition [{tag}]", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log(f"figure -> {path}")


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/hogg18/hogg18_meta.parquet")
    ap.add_argument("--results", default=None,
                    help="run_full_gadi.py results parquet (plx_sp/err_sp); "
                         "omit to score the paper's published spec_parallax only")
    ap.add_argument("--pm-cache", default=None,
                    help="proper-motion cache parquet (default: next to --meta)")
    ap.add_argument("--allstar-raw", default=None,
                    help="DR14 allStar-l31c.2.fits for the FPARAM logg fallback "
                         "(default: next to --meta)")
    ap.add_argument("--fig", default="hogg18_clusters.png")
    args = ap.parse_args()

    meta = pd.read_parquet(args.meta)
    cache = args.pm_cache or os.path.join(os.path.dirname(args.meta),
                                          "cluster_pm_cache.parquet")

    # candidates = union of the sky cones; fetch PMs only for those
    ra, dec = meta["ra"].to_numpy(float), meta["dec"].to_numpy(float)
    cand = np.zeros(len(meta), bool)
    for c in CLUSTERS.values():
        cand |= np.hypot((ra - c["ra"]) * np.cos(np.radians(c["dec"])),
                         dec - c["dec"]) < c["radius"]
    ok = cand & np.isfinite(meta["source_id"].to_numpy(float))
    log(f"{int(cand.sum())} stars in the three cones, {int(ok.sum())} with a Gaia match")
    pm = fetch_proper_motions(meta["source_id"].to_numpy()[ok], cache)
    members = select_members(meta, pm)
    logg = member_logg(meta, args.allstar_raw or os.path.join(
        os.path.dirname(args.meta), "allStar-l31c.2.fits"))

    # paper's own predictions (Fig. 4 as published)
    report(meta, members,
           meta["spec_parallax"].to_numpy(float),
           meta["spec_parallax_err"].to_numpy(float), tag="paper", logg=logg)

    if args.results:
        res = pd.read_parquet(args.results,
                              columns=["sdss_id", "plx_sp", "err_sp"])
        m = meta[["sdss_id"]].merge(res, on="sdss_id", how="left")
        plx_sp = m["plx_sp"].to_numpy(float)
        err_sp = m["err_sp"].to_numpy(float)
        report(meta, members, plx_sp, err_sp, tag="ours", logg=logg)
        fig4(meta, members, plx_sp, err_sp, logg, "ours", args.fig)
    else:
        fig4(meta, members,
             meta["spec_parallax"].to_numpy(float),
             meta["spec_parallax_err"].to_numpy(float), logg, "paper", args.fig)


if __name__ == "__main__":
    main()
