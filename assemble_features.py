#!/usr/bin/env python3
"""
assemble_features.py  —  build the (N, 7405) spectral matrix + (N, 8) photometry
array for the Hogg+18 spectrophotometric-parallax model, keyed to the 2MASS IDs
in the published Zenodo catalog.

Data sources (as chosen):
  * APOGEE DR14 aspcapStar spectra  -> SDSS SAS over HTTP
  * Gaia DR2 parallax + G/BP/RP, 2MASS JHK, WISE W1/W2
        -> Gaia archive, via the PRECOMPUTED crossmatches
           (gaiadr2.tmass_best_neighbour, allwise_best_neighbour),
           i.e. the same Marrese+ official match Paper I used.

Run on Gadi (needs disk for the spectra + outbound HTTP to SAS & ESA).
This is a SCAFFOLD: it is structured into resumable stages so a 44k-spectrum
pull can be checkpointed and re-run. Tune POOL / paths for your allocation.

Stages:
  1. crossmatch  : ADQL join -> photometry+parallax table (one query, fast)
  2. fetch       : download aspcapStar FITS for each 2MASS ID (the slow part)
  3. build       : read flux HDUs, mask to the shared good-pixel set,
                   pseudo-continuum is ALREADY applied in aspcapStar HDU1,
                   assemble X = [1, 8 mags, ln(flux_1..L)] and y = (plx, err)

Outputs (npz, joinable back to catalog by 2MASS_ID):
  features.npz : ids, X_phot (N,8), X_spec (N,L), good_pixel_mask (Lfull,)
  targets.npz  : ids, plx_a, err_a   (Gaia DR2, NOT offset-adjusted here)
"""
from __future__ import annotations
import os, sys, time, json, gzip, io
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
CATALOG   = "hogg2018.fits"                 # the Zenodo file (source of 2MASS IDs)
WORK      = Path(os.environ.get("SPPHOT_WORK", "./spphot_data"))
SPEC_DIR  = WORK / "aspcapStar"
XMATCH    = WORK / "xmatch.fits"
FEATURES  = WORK / "features.npz"
TARGETS   = WORK / "targets.npz"
POOL      = int(os.environ.get("SPPHOT_POOL", "16"))   # parallel downloads

# DR14 ASPCAP results version
SAS_BASE  = "https://data.sdss.org/sas/dr14/apogee/spectro/aspcap/r8/l31c/l31c.2"
ALLSTAR   = f"{SAS_BASE}/allStar-l31c.2.fits"          # for LOCATION_ID -> file path

# Paper I photometry feature order (Eq. 11): G, BP, RP, J, H, K, W1, W2
PHOT_COLS = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
             "j_m", "h_m", "ks_m", "w1mpro", "w2mpro"]


# ----------------------------------------------------------------------
# stage 1 — crossmatch via precomputed neighbour tables
# ----------------------------------------------------------------------
def stage_crossmatch():
    """One ADQL query: 2MASS ID -> Gaia DR2 + 2MASS + WISE photometry & parallax.

    Uploads the catalog's 2MASS IDs as a user table and joins through the
    precomputed neighbour tables. The 2MASS designation in the Zenodo file is
    the full 'JHHMMSSss+DDMMSSs' string; tmass_best_neighbour stores it WITHOUT
    the leading 'J', so we strip it on upload.

    Two paths:
      * PRIMARY (login): async upload-join — fast, single query, needs a free
        ESA Gaia account. Set GAIA_USER / GAIA_PASS env vars for batch jobs.
      * FALLBACK (anonymous): batched inline-IN. The public endpoint rate-limits
        / 500s on large IN-lists, so batches are small with retry+backoff.
        Verified-correct join; just slower and flakier than the login path.
    """
    from astroquery.gaia import Gaia
    from astropy.io import fits
    from astropy.table import Table, vstack

    ids = [str(s).strip() for s in fits.open(CATALOG)[1].data["2MASS_ID"]]
    keys = [i[2:] if i.startswith("2M") else i for i in ids]

    select = """SELECT xm.original_ext_source_id AS tmass_key,
           g.source_id, g.parallax, g.parallax_error,
           g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
           g.visibility_periods_used,
           g.astrometric_chi2_al, g.astrometric_n_good_obs_al,
           tm.j_m, tm.h_m, tm.ks_m, aw.w1mpro, aw.w2mpro"""
    joins = """
    FROM gaiadr2.tmass_best_neighbour AS xm
    JOIN gaiadr2.gaia_source AS g ON g.source_id = xm.source_id
    JOIN gaiadr1.tmass_original_valid AS tm ON tm.tmass_oid = xm.tmass_oid
    LEFT JOIN gaiadr2.allwise_best_neighbour AS awx ON awx.source_id = g.source_id
    LEFT JOIN gaiadr1.allwise_original_valid AS aw ON aw.allwise_oid = awx.allwise_oid"""

    user = os.environ.get("GAIA_USER")
    if user:  # ---- PRIMARY: authenticated upload-join ----
        Gaia.login(user=user, password=os.environ.get("GAIA_PASS"))
        up = Table({"tmass_key": keys, "orig_id": ids})
        up_path = WORK / "ids_upload.xml"
        up.write(up_path, format="votable", overwrite=True)
        adql = (select.replace("xm.original_ext_source_id AS tmass_key",
                               "u.orig_id, xm.original_ext_source_id AS tmass_key")
                + joins
                + "\n    JOIN tap_upload.ids AS u ON u.tmass_key = xm.original_ext_source_id")
        # simpler: put the upload join first
        adql = f"""{select}, u.orig_id {joins}
        JOIN tap_upload.ids AS u ON u.tmass_key = xm.original_ext_source_id"""
        res = Gaia.launch_job_async(adql, upload_resource=str(up_path),
                                    upload_table_name="ids").get_results()
        res.write(XMATCH, overwrite=True)
        print(f"[xmatch:login] {len(res)}/{len(ids)} matched -> {XMATCH}")
        return res

    # ---- FALLBACK: anonymous batched inline-IN with retry ----
    import time
    BATCH = int(os.environ.get("SPPHOT_XMATCH_BATCH", "200"))
    parts = []
    for i in range(0, len(keys), BATCH):
        chunk = keys[i:i + BATCH]
        inlist = ",".join(f"'{k}'" for k in chunk)
        adql = f"{select}{joins}\n    WHERE xm.original_ext_source_id IN ({inlist})"
        for attempt in range(4):
            try:
                parts.append(Gaia.launch_job_async(adql).get_results())
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[xmatch] batch {i} failed: {repr(e)[:80]}")
                time.sleep(5 * (attempt + 1))
        if (i // BATCH) % 10 == 0:
            print(f"[xmatch] {i+len(chunk)}/{len(keys)}")
        time.sleep(1)
    res = vstack(parts)
    # restore full '2M...' orig_id from the stripped key
    res["orig_id"] = ["2M" + k for k in res["tmass_key"]]
    res.write(XMATCH, overwrite=True)
    print(f"[xmatch:anon] {len(res)}/{len(ids)} matched -> {XMATCH}")
    return res


# ----------------------------------------------------------------------
# stage 2 — fetch aspcapStar spectra from SAS
# ----------------------------------------------------------------------
def _aspcap_url(loc_id, twomass_id):
    """DR14 aspcapStar path. loc_id from allStar; '2M'+id is the file token.
    For the main survey (apo25m): .../<LOCATION_ID>/aspcapStar-r8-<ID>.fits
    Telescope field uses 'apo1m'/healpix layout — handle via allStar FIELD."""
    tok = twomass_id if twomass_id.startswith("2M") else "2M" + twomass_id
    return f"{SAS_BASE}/apo25m/{loc_id}/aspcapStar-r8-{tok}.fits"


def stage_fetch(xmatch_tbl=None):
    """Download aspcapStar FITS for each matched star. Resumable: skips files
    already on disk. Needs LOCATION_ID/FIELD from allStar — fetch that once.

    NOTE: a minority of targets sit in apo1m fields with a different path; the
    robust approach is to read LOCATION_ID & FIELD & FILE from allStar rather
    than reconstruct URLs. Stub below shows the structure; wire allStar lookup
    to your local copy if you have one (faster than re-deriving).
    """
    import requests
    from astropy.io import fits
    from concurrent.futures import ThreadPoolExecutor, as_completed

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    if xmatch_tbl is None:
        xmatch_tbl = fits.open(XMATCH)[1].data
    ids = [str(s).strip() for s in xmatch_tbl["orig_id"]]

    # --- map 2MASS ID -> aspcapStar URL via allStar (download once) ---
    allstar_local = WORK / "allStar-l31c.2.fits"
    if not allstar_local.exists():
        print("[fetch] downloading allStar summary (large, ~1.5 GB)...")
        _stream_download(ALLSTAR, allstar_local)
    a = fits.open(allstar_local)[1].data
    apo_id = np.array([str(s).strip() for s in a["APOGEE_ID"]])
    loc    = np.array([str(s).strip() for s in a["LOCATION_ID"]])
    fld    = np.array([str(s).strip() for s in a["FIELD"]])
    tele   = np.array([str(s).strip() for s in a["TELESCOPE"]])
    lookup = {aid: (l, f, t) for aid, l, f, t in zip(apo_id, loc, fld, tele)}

    def url_for(tmass):
        meta = lookup.get(tmass) or lookup.get("2M" + tmass)
        if meta is None:
            return None
        loc_id, field, telescope = meta
        tok = tmass if tmass.startswith("2M") else "2M" + tmass
        # DR14 layout: aspcap/r8/l31c/l31c.2/<TELESCOPE>/<FIELD>/aspcapStar-r8-<ID>.fits
        return f"{SAS_BASE}/{telescope}/{field}/aspcapStar-r8-{tok}.fits"

    def one(tmass):
        out = SPEC_DIR / f"aspcapStar-{tmass}.fits"
        if out.exists() and out.stat().st_size > 0:
            return tmass, "skip"
        u = url_for(tmass)
        if u is None:
            return tmass, "no-meta"
        try:
            r = requests.get(u, timeout=60)
            if r.status_code == 200:
                out.write_bytes(r.content)
                return tmass, "ok"
            return tmass, f"http-{r.status_code}"
        except Exception as e:
            return tmass, f"err-{type(e).__name__}"

    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=POOL) as ex:
        futs = {ex.submit(one, t): t for t in ids}
        for i, fut in enumerate(as_completed(futs)):
            _, status = fut.result()
            if status == "ok": n_ok += 1
            elif status == "skip": n_skip += 1
            else: n_fail += 1
            if (i + 1) % 500 == 0:
                print(f"[fetch] {i+1}/{len(ids)}  ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"[fetch] done. ok={n_ok} skip={n_skip} fail={n_fail}")


def _stream_download(url, dest, chunk=1 << 20):
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for c in r.iter_content(chunk):
                f.write(c)


# ----------------------------------------------------------------------
# stage 3 — build feature matrix
# ----------------------------------------------------------------------
def stage_build(xmatch_tbl=None):
    """Read aspcapStar HDU1 (already pseudo-continuum-normalized), build the
    shared good-pixel mask across all stars, and assemble X_spec = ln(flux),
    X_phot = the 8 magnitudes, y = (parallax, error).

    Good-pixel rule (matches Paper I intent): keep pixels that have valid,
    nonzero flux in (essentially) all stars; drop the ~109 always-bad ones.
    aspcapStar normalized flux is ~1 in continuum; bad pixels are 0 or NaN.
    """
    from astropy.io import fits
    if xmatch_tbl is None:
        xmatch_tbl = fits.open(XMATCH)[1].data
    ids = [str(s).strip() for s in xmatch_tbl["orig_id"]]

    # first pass: load all fluxes, track which pixels are ever bad
    fluxes, kept_ids, phot_rows, plx_rows = [], [], [], []
    Lfull = None
    bad_accum = None
    for k, tmass in enumerate(ids):
        fp = SPEC_DIR / f"aspcapStar-{tmass}.fits"
        if not fp.exists():
            continue
        try:
            h = fits.open(fp)
            flux = np.asarray(h[1].data, float)   # HDU1 = normalized flux
        except Exception:
            continue
        if Lfull is None:
            Lfull = flux.size
            bad_accum = np.zeros(Lfull, bool)
        if flux.size != Lfull:
            continue
        bad = ~np.isfinite(flux) | (flux <= 0)
        bad_accum |= bad

        row = xmatch_tbl[k]
        phot = np.array([row[c] for c in PHOT_COLS], float)
        if not np.all(np.isfinite(phot)):
            continue   # Paper I requires COMPLETE photometry

        fluxes.append(flux)
        phot_rows.append(phot)
        plx_rows.append([row["parallax"], row["parallax_error"]])
        kept_ids.append(tmass)

    good = ~bad_accum                       # shared good-pixel mask
    L = int(good.sum())
    print(f"[build] full grid={Lfull}, good pixels kept={L} "
          f"(Paper I kept 7405)")

    X_spec = np.empty((len(kept_ids), L), float)
    for i, flux in enumerate(fluxes):
        f = flux[good]
        f = np.where(f > 0, f, 1.0)         # guard residual zeros before log
        X_spec[i] = np.log(f)               # ln(flux), as in Eq. 11
    X_phot = np.asarray(phot_rows, float)
    y      = np.asarray(plx_rows, float)

    np.savez_compressed(FEATURES, ids=np.array(kept_ids),
                        X_phot=X_phot, X_spec=X_spec,
                        good_pixel_mask=good, phot_cols=np.array(PHOT_COLS))
    np.savez_compressed(TARGETS, ids=np.array(kept_ids),
                        plx_a=y[:, 0], err_a=y[:, 1])
    print(f"[build] X_spec {X_spec.shape}, X_phot {X_phot.shape} -> {FEATURES}")
    print(f"[build] targets -> {TARGETS}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("crossmatch", "all"):
        stage_crossmatch()
    if stage in ("fetch", "all"):
        stage_fetch()
    if stage in ("build", "all"):
        stage_build()
