#!/usr/bin/env python3
"""
apply_nn.py — predict spectrophotometric parallaxes/distances for NEW stars with
a heteroscedastic-NN checkpoint saved by train_nn.py, WITHOUT refitting.

The NN analogue of apply_model.py: it reuses run_full_gadi's metadata loader and
streaming spectral builder (projected onto the SAVED good-pixel mask, with the
same telescope lit-masks), so apply-time features and normalization are identical
to training. Features = standardize([phot | ln-flux], saved mu/sd) -> HetMLP ->
(plx_sp, sig_sp); err_sp = sig_sp * std_factor.

Built for the cluster test: it carries a cluster label through to the output, so
the result feeds straight into cluster_test.py. Two ways to label:
  * one file per cluster -> --cluster-name NGC6791
  * a combined file with a per-star name column -> --cluster-col <colname>
    (read from the parquet scalar columns, else from the allStar table)

Gaia parallaxes are written zero-point corrected (plx = plx_raw - zeropoint), the
same convention the model was trained on, so the cluster test compares like with
like (use offset=0.0 there). If the cluster parquet has no 'zeropoint' column,
plx falls back to plx_raw and you should pass the global offset to the test.

Usage:
  python apply_nn.py --model spphot_nn_results_model.pt \
      --parquet ngc6791_spectra.parquet --allstar astraAllStarASPCAP-0.8.0.fits \
      --pixel-mask-dir <dir> --cluster-name NGC6791 --out ngc6791_nn.parquet
"""
from __future__ import annotations
import argparse
import numpy as np

import run_full_gadi as R          # load_metadata, build_lnflux_streaming, load_pixel_masks, log


def _resolve_cluster(parquet, allstar, col, sdss_ids):
    """Cluster-name array aligned to sdss_ids, reading `col` from the parquet
    scalar columns or (failing that) the allStar table."""
    import pyarrow.parquet as pq, pandas as pd
    from astropy.io import fits
    if col in set(pq.ParquetFile(parquet).schema_arrow.names):
        t = pq.read_table(parquet, columns=["sdss_id", col]).to_pandas()
    else:
        a = fits.open(allstar)[2].data
        def native(x):
            x = np.asarray(x)
            return x.astype(x.dtype.newbyteorder("=")) if x.dtype.byteorder == ">" else x
        t = pd.DataFrame({"sdss_id": native(a["sdss_id"]), col: native(a[col])}
                         ).drop_duplicates("sdss_id")
    lut = dict(zip(t["sdss_id"].to_numpy(), t[col].to_numpy()))
    return np.array([lut.get(s) for s in sdss_ids], dtype=object)


def load_bundle(model_path, pixel_mask_dir=None, device="auto"):
    """Load a train_nn.py checkpoint + the telescope lit-masks ONCE, into a reusable
    bundle. Build it once and pass it to apply_to_parquet() for every cluster so the
    (expensive) checkpoint load and HetMLP construction are not repeated per file."""
    import torch
    from train_nn import HetMLP

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # weights_only=False: our own train_nn.py checkpoint stores numpy arrays
    # (good_pixel_mask, mu/sd) that PyTorch 2.6's restricted unpickler rejects.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    good = np.asarray(ckpt["good_pixel_mask"], bool)
    mu = np.asarray(ckpt["mu"], np.float32)
    sd = np.asarray(ckpt["sd"], np.float32)
    std_factor = float(ckpt.get("std_factor", 1.0))
    label_cols = list(ckpt["label_cols"])
    hidden = tuple(ckpt["hidden"])
    dropout = float(ckpt.get("dropout", 0.0))
    d_in = int(ckpt["d_in"])
    R.log(f"loaded NN: good pixels={int(good.sum())}, d_in={d_in}, hidden={hidden}, "
          f"std_factor={std_factor:.3f}, loss={ckpt.get('loss')}, device={device}")

    model = HetMLP(d_in, hidden, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "model": model, "good": good, "mu": mu, "sd": sd,
        "std_factor": std_factor, "label_cols": label_cols, "d_in": d_in,
        "tel_masks": R.load_pixel_masks(pixel_mask_dir), "device": device,
    }


def apply_to_parquet(bundle, parquet, allstar, *, out=None, cluster_name=None,
                     cluster_col=None, f_max=2.0, batch_rows=20000):
    """Infer plx_sp/err_sp for one spectra parquet with a preloaded `bundle`.

    Mirrors training feature assembly exactly (saved pixel mask + telescope lit
    masks + saved standardization). Writes `out` if given and returns the result
    DataFrame so a caller can concatenate clusters for the test without re-reading.
    """
    import pandas as pd
    from train_nn import predict_nn

    model, good, mu, sd = bundle["model"], bundle["good"], bundle["mu"], bundle["sd"]
    std_factor, label_cols, d_in = bundle["std_factor"], bundle["label_cols"], bundle["d_in"]
    tel_masks, device = bundle["tel_masks"], bundle["device"]

    merged, n_parquet = R.load_metadata(parquet, allstar)

    phot_all = merged[label_cols].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    R.log(f"keep (complete photometry): {int(keep.sum())} / {n_parquet}")

    # spectra on the SAVED pixel mask + telescope lit masks (identical to training)
    X_spec, _, star_bad = R.build_lnflux_streaming(
        parquet, keep, f_max=f_max, batch_rows=batch_rows,
        fixed_good=good, tel_masks=tel_masks)

    feats = np.hstack([phot_all[keep].astype(np.float32), X_spec]).astype(np.float32, copy=False)
    assert feats.shape[1] == d_in, (
        f"feature dim {feats.shape[1]} != model d_in {d_in} "
        f"(pixel-mask / photometry mismatch with training)")

    R.log("predicting with the all-train model ...")
    plx_sp, sig_sp = predict_nn(model, feats, np.arange(feats.shape[0]), mu, sd, device)
    err_sp = sig_sp * std_factor
    dist_kpc = 1.0 / plx_sp

    # Gaia targets, zero-point corrected like training (NaN where no zeropoint)
    plx_raw = merged["plx"].to_numpy(float)[keep]
    zpt = merged["zeropoint"].to_numpy(float)[keep]
    plx_corr = plx_raw - zpt
    n_zpt = int(np.isfinite(zpt).sum())
    if n_zpt == 0:
        R.log("no zeropoint in parquet -> plx = plx_raw; pass the global offset to "
              "the cluster test")
        plx_corr = plx_raw
    ids_k = merged["sdss_id"].to_numpy()[keep]

    cols = {
        "sdss_id": ids_k,
        "plx": plx_corr, "e_plx": merged["e_plx"].to_numpy(float)[keep],
        "plx_raw": plx_raw, "zeropoint": zpt,
        "plx_sp": plx_sp, "err_sp": err_sp, "dist_sp_kpc": dist_kpc,
        "r_med_photogeo_pc": merged["r_med_photogeo"].to_numpy(float)[keep],
        "spec_bad_frac": star_bad,
    }
    if cluster_col:
        cols["cluster"] = _resolve_cluster(parquet, allstar, cluster_col, ids_k)
    elif cluster_name:
        cols["cluster"] = np.full(int(keep.sum()), cluster_name)

    df = pd.DataFrame(cols)
    if out is not None:
        df.to_parquet(out, index=False)
        R.log(f"wrote {out}  ({int(keep.sum())} stars"
              + (f", {len(set(cols['cluster']))} clusters)" if "cluster" in cols else ")"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt checkpoint saved by train_nn.py")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--pixel-mask-dir", default=None,
                    help="same pixel-mask dir used in training (telescope lit masks)")
    ap.add_argument("--out", default="applied_nn.parquet")
    ap.add_argument("--cluster-name", default=None,
                    help="label every star with this cluster name (file == one cluster)")
    ap.add_argument("--cluster-col", default=None,
                    help="per-star cluster-name column to carry through (combined file)")
    ap.add_argument("--f-max", type=float, default=2.0,
                    help="bad-pixel flux clip; must match training (default 2.0)")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    bundle = load_bundle(args.model, args.pixel_mask_dir, args.device)
    apply_to_parquet(bundle, args.parquet, args.allstar, out=args.out,
                     cluster_name=args.cluster_name, cluster_col=args.cluster_col,
                     f_max=args.f_max, batch_rows=args.batch_rows)


if __name__ == "__main__":
    main()
