"""
cluster_test.py — external cluster validation for spectrophotometric parallaxes.

This is the Hogg, Eilers & Rix (2018) Figure-4 test: cluster members are all at
ONE true distance, so their spectrophotometric parallaxes should (a) form a tight
clump and (b) have an inverse-variance-weighted mean that agrees with the Gaia
cluster mean (and with the literature distance). Two things make it the strongest
validation we have:

  * It is EXTERNAL: membership comes from a published catalog (proper motion + sky
    + RV), not from the Gaia parallax of each star, so it probes the regime where
    Gaia is uninformative.
  * The member-to-member spread gives a Gaia-INDEPENDENT calibration check. Since
    every member shares one true parallax, the scatter of plx_sp about the cluster
    mean must be explained by err_sp alone. If the internal chi2 >> 1, err_sp is
    underestimated (overconfident) — corroborating the calibration finding from
    spphot_eval.calibration_bins() WITHOUT using any Gaia parallax.

The scoring functions take only parallax-space arrays, exactly like spphot_eval,
so they work on the published Zenodo catalog and on your NN parquet alike.

Membership is the only fiddly part because your catalog is keyed by sdss_id while
the member catalogs are keyed by 2MASS / Gaia source_id. See match_membership()
and add_crosswalk() for the two-step (load members -> crosswalk ids -> match).

In SDSS-V astra allStar the 2MASS id (sdss4_apogee_id) is populated for only
~8% of stars, but gaia_dr3_source_id is there for ~99.6%. So crosswalk on the
GAIA source id and use Gaia-keyed member catalogs: Cantat-Gaudin et al. 2020
(open clusters, Gaia DR2 ids) and Vasiliev & Baumgardt 2021 (globulars, Gaia
EDR3 ids). Match each catalog to the corresponding allStar column
(gaia_dr2_source_id vs gaia_dr3_source_id).

Typical use (NN parquet + Cantat-Gaudin open clusters):

    import pandas as pd, cluster_test as C
    df = pd.read_parquet("nn_out.parquet")
    cg = C.load_cantat_gaudin("cantat-gaudin-2020-members.fits", prob_min=0.7)
    df = C.add_crosswalk(df, "astraAllStarASPCAP-0.8.0.fits",     # add df['gaia_id']
                         xref_left="sdss_id", xref_right="gaia_dr2_source_id",
                         new_col="gaia_id")                       # DR2 to match CG2020
    members = C.match_membership(df, cg, cat_id="gaia_id",
                                 mem_cluster="cluster", mem_id="gaia_id")
    report = C.run_cluster_test(df, members, offset=0.0)
    C.print_cluster_test(report)
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# literature parallaxes (mas) — OPTIONAL external truth, EDIT/VERIFY these.
# ----------------------------------------------------------------------
# These are approximate (1000/distance_pc) for common APOGEE clusters and the
# three Hogg+18 used. They are only a sanity reference; the core metrics below
# do not need them. Replace with your preferred source (Cantat-Gaudin et al.
# 2020 for open clusters, Baumgardt & Vasiliev 2021 for globulars) before
# quoting anything. NGC 2682 == M67; the Hogg+18 Fig-4 "NGC2862" is almost
# certainly M67 (its ~1.1 mas line matches M67's distance).
LIT_PLX_MAS = {
    "M67":       1.17,   # NGC 2682, ~850 pc
    "NGC2682":   1.17,
    "NGC6791":   0.23,   # ~4.3 kpc
    "NGC6819":   0.42,   # ~2.4 kpc
    "NGC7789":   0.50,   # ~2.0 kpc
    "M71":       0.25,   # NGC 6838, ~4.0 kpc (Hogg+18)
    "NGC6838":   0.25,
    "M107":      0.156,  # NGC 6171, ~6.4 kpc (Hogg+18)
    "NGC6171":   0.156,
}


# ----------------------------------------------------------------------
# core metrics (parallax space only — no inversion to distance)
# ----------------------------------------------------------------------
def ivw_mean(vals, errs):
    """Inverse-variance-weighted mean and its formal error. NaNs dropped."""
    vals, errs = np.asarray(vals, float), np.asarray(errs, float)
    m = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
    if m.sum() == 0:
        return np.nan, np.nan
    w = 1.0 / errs[m]**2
    mean = float(np.sum(w * vals[m]) / np.sum(w))
    return mean, float(np.sqrt(1.0 / np.sum(w)))


def _robust_scatter(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.48 * np.median(np.abs(x - np.median(x)))


def cluster_metrics(plx_sp, err_sp, plx_a, err_a, *, offset=0.0,
                    lit_plx=None, err_scale=1.0):
    """All metrics for ONE cluster's members.

    plx_sp/err_sp : spectrophotometric parallax + quoted error of each member
    plx_a/err_a   : Gaia parallax + error of each member (for the mean comparison)
    offset        : Gaia zero-point offset to add to plx_a before comparing
    lit_plx       : optional literature parallax [mas] for an external truth check
    err_scale     : multiply err_sp by this before scoring (try your recal factor c
                    to see if the internal chi2 drops to ~1)

    Returns a dict. The headline diagnostics:
      sp_mean +/- sp_mean_err : IVW spec parallax of the cluster (Fig-4 vertical line)
      gaia_mean +/- ...       : IVW Gaia parallax (the other Fig-4 line)
      sp_vs_gaia_sigma        : (sp_mean - gaia_mean - offset)/combined_err  -> bias
      tightness_frac          : robust scatter of members / sp_mean  -> clumpiness
      internal_chi2 / _robust : member spread normalized by err_sp.  GAIA-INDEPENDENT
                                calibration: >>1 means err_sp is too small.
    """
    plx_sp = np.asarray(plx_sp, float)
    err_sp = np.asarray(err_sp, float) * err_scale
    plx_a = np.asarray(plx_a, float)
    err_a = np.asarray(err_a, float)

    sp_mean, sp_mean_err = ivw_mean(plx_sp, err_sp)
    gaia_mean, gaia_mean_err = ivw_mean(plx_a, err_a)

    # member-to-member coherence about the common spec mean (one true parallax)
    good = np.isfinite(plx_sp) & np.isfinite(err_sp) & (err_sp > 0)
    n = int(good.sum())
    chi = (plx_sp[good] - sp_mean) / err_sp[good]
    internal_chi2 = float(np.mean(chi**2)) if n else np.nan
    internal_robust = float(_robust_scatter(chi)**2) if n else np.nan
    tight = _robust_scatter(plx_sp[good])

    # spec vs Gaia mean (the two vertical lines in Fig 4)
    comb = np.sqrt(sp_mean_err**2 + gaia_mean_err**2)
    sp_vs_gaia_sigma = ((sp_mean - gaia_mean - offset) / comb
                        if np.isfinite(comb) and comb > 0 else np.nan)

    rep = {
        "n": n,
        "sp_mean": sp_mean, "sp_mean_err": sp_mean_err,
        "gaia_mean": gaia_mean, "gaia_mean_err": gaia_mean_err,
        "sp_vs_gaia_fracdiff": (sp_mean - gaia_mean - offset) / gaia_mean
                               if np.isfinite(gaia_mean) and gaia_mean != 0 else np.nan,
        "sp_vs_gaia_sigma": sp_vs_gaia_sigma,
        "tightness_mas": float(tight),
        "tightness_frac": float(tight / sp_mean) if sp_mean else np.nan,
        "internal_chi2": internal_chi2,
        "internal_robust_chi2": internal_robust,
    }
    if lit_plx is not None and np.isfinite(lit_plx):
        rep["lit_plx"] = float(lit_plx)
        rep["sp_vs_lit_fracdiff"] = (sp_mean - lit_plx) / lit_plx
    return rep


# ----------------------------------------------------------------------
# membership matching  (cat keyed by sdss_id; members keyed by 2MASS/Gaia id)
# ----------------------------------------------------------------------
def add_crosswalk(df, xref, *, left_id="sdss_id", xref_left="sdss_id",
                  xref_right="gaia_dr3_source_id", new_col="gaia_id", hdu=None):
    """Attach an external id to df via a crosswalk table (e.g. allStar).

    df    : your catalog (a pandas DataFrame with `left_id`)
    xref  : a crosswalk -- a pandas DataFrame, or a path to a FITS/parquet with
            both `xref_left` (matches df.left_id) and `xref_right` (the id you
            want, e.g. gaia_dr2_source_id / gaia_dr3_source_id).
    hdu   : FITS extension to read; None -> first HDU with rows (astra allStar
            keeps its table in HDU 2, not 1).
    Adds df[new_col]. Left-join, so unmatched rows get NaN. Only the two needed
    columns are read from FITS, so a multi-GB allStar costs ~tens of MB.
    """
    import pandas as pd, numpy as np
    if isinstance(xref, str):
        if xref.endswith((".fits", ".fit", ".fits.gz")):
            from astropy.io import fits
            with fits.open(xref, memmap=True) as f:
                if hdu is None:
                    hdu = next(i for i, h in enumerate(f)
                               if h.header.get("NAXIS2", 0) > 0)
                d = f[hdu].data
                # FITS is big-endian; pandas/numpy hashing wants native order
                native = lambda a: a.astype(a.dtype.newbyteorder("="))
                xref = pd.DataFrame(
                    {left_id: native(np.asarray(d[xref_left])),
                     new_col: native(np.asarray(d[xref_right]))})
        else:
            xref = pd.read_parquet(xref, columns=[xref_left, xref_right]).rename(
                columns={xref_left: left_id, xref_right: new_col})
    else:
        xref = xref[[xref_left, xref_right]].rename(
            columns={xref_left: left_id, xref_right: new_col})
    # drop unmatched/zero ids so they can't collide on a 0 sentinel
    if np.issubdtype(np.asarray(xref[new_col]).dtype, np.integer):
        xref = xref[xref[new_col] != 0]
        # nullable Int64 so the left-join's unmatched rows become <NA> WITHOUT
        # upcasting to float64 -- 19-digit Gaia source ids exceed float precision
        xref[new_col] = xref[new_col].astype("Int64")
    xref = xref.drop_duplicates(left_id)
    return df.merge(xref, on=left_id, how="left")


def members_from_labels(df, col="cluster"):
    """{cluster: row-position array} from a per-star cluster-label column already
    IN df. Use this when the catalog was crossmatched to clusters upstream (e.g.
    apply_nn.py --cluster-name / --cluster-col), so no id matching is needed.
    """
    import pandas as pd
    labels = df[col].to_numpy()
    out = {}
    for cl in pd.unique(labels):
        if cl is None or (isinstance(cl, float) and np.isnan(cl)):
            continue
        out[str(cl)] = np.where(labels == cl)[0]
    return out


def match_membership(df, members, *, cat_id="sdss_id", mem_cluster="cluster",
                     mem_id=None, normalize_ids=False):
    """Map each cluster to the integer row positions of its members in df.

    df         : your catalog DataFrame
    members    : either a dict {cluster_name: iterable of member ids}, or a
                 DataFrame with a cluster column (`mem_cluster`) and an id column
                 (`mem_id`).
    cat_id     : the df column to match on (e.g. 'sdss_id', or 'tmass_id' after
                 add_crosswalk).
    normalize_ids : strip/upper-case and drop a leading '2M' on both sides
                    (use for messy 2MASS string ids).

    Returns {cluster_name: np.ndarray of integer positions into df}. Clusters
    with zero matches are dropped (with no error).
    """
    import pandas as pd

    def key(v):
        """Common, exact key for int (Gaia source) ids and string (2MASS) ids."""
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return None
        if normalize_ids:
            return str(v).strip().upper().replace("2M", "")
        if isinstance(v, (int, np.integer)):      # python int and np.int64 -> same key
            return int(v)
        return str(v).strip()

    # build id -> row-position lookup (first occurrence wins). dtype=object keeps
    # nullable Int64 ids as exact python ints -- a plain .to_numpy() would cast
    # an Int64-with-NA column to float64 and corrupt 19-digit Gaia source ids.
    lut = {}
    for i, v in enumerate(df[cat_id].to_numpy(dtype=object, na_value=None)):
        k = key(v)
        if k is not None:
            lut.setdefault(k, i)

    if not isinstance(members, dict):
        mem_df = members
        members = {c: mem_df.loc[mem_df[mem_cluster] == c, mem_id].to_numpy()
                   for c in sorted(mem_df[mem_cluster].unique())}

    out = {}
    for cl, ids in members.items():
        idx = [lut[key(v)] for v in np.asarray(ids) if key(v) in lut]
        if idx:
            out[cl] = np.array(sorted(set(idx)))
    return out


# ----------------------------------------------------------------------
# member-catalog loaders  (thin; verify column names against your file)
# ----------------------------------------------------------------------
def load_occam(path, *, prob_min=0.8, cluster_col="CLUSTER",
               id_col="APOGEE_ID", prob_col="CG_PROB"):
    """OCCAM open-cluster members -> DataFrame[cluster, tmass_id].

    OCCAM (Open Cluster Chemical Abundances & Mapping) ships membership
    probabilities; keep prob >= prob_min. APOGEE_ID is the 2MASS-style id. Adjust
    the column names to your OCCAM file version (DR16/DR17 differ slightly).
    """
    from astropy.table import Table
    t = Table.read(path)
    df = t[[cluster_col, id_col, prob_col]].to_pandas()
    df = df[df[prob_col] >= prob_min]
    return df.rename(columns={cluster_col: "cluster", id_col: "tmass_id"})[
        ["cluster", "tmass_id"]]


def load_cantat_gaudin(path, *, prob_min=0.7, cluster_col="Cluster",
                       id_col="GaiaDR2", prob_col="proba"):
    """Cantat-Gaudin et al. (2020) open-cluster members -> DataFrame[cluster,
    gaia_id]. Keyed by Gaia DR2 source_id, so crosswalk df to gaia_dr2_source_id.
    `proba` is the membership probability (keep >= prob_min). Column names follow
    the VizieR table (J/A+A/640/A1); adjust if your download differs.
    """
    from astropy.table import Table
    t = Table.read(path)
    df = t[[cluster_col, id_col, prob_col]].to_pandas()
    df = df[df[prob_col] >= prob_min]
    return df.rename(columns={cluster_col: "cluster", id_col: "gaia_id"})[
        ["cluster", "gaia_id"]]


def load_vasiliev_baumgardt(path, *, prob_min=0.9, cluster_col="Cluster",
                            id_col="source_id", prob_col="memberprob"):
    """Vasiliev & Baumgardt (2021) globular-cluster members -> DataFrame[cluster,
    gaia_id]. Keyed by Gaia source_id, so crosswalk df to source_id first.
    """
    from astropy.table import Table
    t = Table.read(path)
    df = t[[cluster_col, id_col, prob_col]].to_pandas()
    df = df[df[prob_col] >= prob_min]
    return df.rename(columns={cluster_col: "cluster", id_col: "gaia_id"})[
        ["cluster", "gaia_id"]]


# ----------------------------------------------------------------------
# top-level driver
# ----------------------------------------------------------------------
def run_cluster_test(df, members, *, offset=0.0, err_scale=1.0,
                     plx="plx_sp", err="err_sp", plx_g="plx", err_g="e_plx",
                     lit=LIT_PLX_MAS, min_members=5):
    """Run cluster_metrics for every matched cluster.

    df       : catalog DataFrame
    members  : {cluster: row-position array} from match_membership()
    columns  : plx/err are the spec columns; plx_g/err_g the Gaia columns.
    err_scale: pass your recal factor c to preview calibrated internal_chi2.
    min_members : skip clusters with fewer matched members.

    Returns {cluster: metrics-dict}, sorted by member count.
    """
    P, E = df[plx].to_numpy(), df[err].to_numpy()
    PA, EA = df[plx_g].to_numpy(), df[err_g].to_numpy()
    out = {}
    for cl, idx in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < min_members:
            continue
        key = str(cl).upper().replace(" ", "")
        out[cl] = cluster_metrics(P[idx], E[idx], PA[idx], EA[idx],
                                  offset=offset, lit_plx=lit.get(key),
                                  err_scale=err_scale)
    return out


def print_cluster_test(report):
    if not report:
        print("cluster test: no clusters matched (check id crosswalk / min_members)")
        return
    print("cluster test (spectrophotometric parallax vs Gaia + internal coherence)")
    print(f"  {'cluster':<10} {'N':>4} {'sp_mean':>8} {'gaia':>8} "
          f"{'Δsp/gaia':>9} {'σ':>6} {'tight%':>7} {'intχ²':>6} {'Δlit%':>7}")
    intchis = []
    for cl, r in report.items():
        lit = f"{100*r['sp_vs_lit_fracdiff']:+6.1f}" if "sp_vs_lit_fracdiff" in r else "    -"
        print(f"  {str(cl):<10} {r['n']:>4} {r['sp_mean']:>8.3f} {r['gaia_mean']:>8.3f} "
              f"{100*r['sp_vs_gaia_fracdiff']:>+8.1f}% {r['sp_vs_gaia_sigma']:>+5.1f} "
              f"{100*r['tightness_frac']:>6.1f}% {r['internal_chi2']:>6.1f} {lit:>6}")
        if np.isfinite(r["internal_chi2"]):
            intchis.append(r["internal_chi2"])
    print("  ---")
    print("  Δsp/gaia : fractional offset of the two IVW means (bias; ~0 is good)")
    print("  σ        : that offset in sigma (|σ|<~2-3 is consistent)")
    print("  tight%   : member robust scatter / mean (real distance spread)")
    print("  intχ²    : member spread / quoted err_sp.  GAIA-FREE calibration:")
    if intchis:
        print(f"             median across clusters = {np.median(intchis):.1f} "
              f"(>>1 -> err_sp overconfident; sqrt = the recal factor)")
