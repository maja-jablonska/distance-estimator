# legacy/ — superseded scripts

Nothing here is imported by the live pipeline, invoked by a PBS job, or used by a
notebook. Kept for reference.

| script | what it was | superseded by |
|---|---|---|
| `assemble_features.py` (+ `.pbs`) | DR14 web-ETL: ADQL crossmatch (Gaia DR2 + 2MASS + WISE) + aspcapStar downloads from SDSS SAS → `features.npz` | the parquet workflow (`spectra_infer_parallax_zpt.parquet` + `astraAllStarASPCAP` FITS) |
| `build_from_parquet.py` | non-streaming spectral-matrix builder (vstacks everything; ~165 GB at 800k stars) | `spphot.data.build_lnflux_streaming` |
| `apply_model.py` | apply a linear `.npz` checkpoint to new stars | the NN path (`apply_nn.py`); still runnable: `PYTHONPATH=. python legacy/apply_model.py ...` from the repo root |
| `check_pixel_flags.py` | one-off diagnostic: do spectra share a common per-telescope pixel_flags mask? | `build_pixel_mask.py` (the answer became the lit-pixel masks) |
