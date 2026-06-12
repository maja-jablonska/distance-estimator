"""Check whether all spectra share a single common 'pixel_flags' mask per telescope:
one mask for every spectrum found in hdus[3] (APO), and one for every spectrum
in hdus[4] (LCO). Empty HDUs (shape (0, 0)) are ignored.

Usage: python check_pixel_flags.py <dir_or_glob> [more dirs/globs...]
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits


def collect_files(args):
    files = []
    for arg in args:
        if os.path.isdir(arg):
            files.extend(glob.glob(os.path.join(arg, "**", "*.fits*"), recursive=True))
        else:
            files.extend(glob.glob(arg))
    return sorted(set(files))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    files = collect_files(sys.argv[1:])
    if not files:
        sys.exit("No FITS files found")

    # per HDU index: reference mask, the file it came from, counts
    ref = {3: None, 4: None}
    ref_file = {3: None, 4: None}
    n_spectra = {3: 0, 4: 0}
    n_diff = {3: 0, 4: 0}
    n_skip = 0

    for path in files:
        try:
            with fits.open(path) as hdus:
                for idx in (3, 4):
                    data = hdus[idx].data
                    if data is None or data["pixel_flags"].size == 0:
                        continue
                    for mask in np.atleast_2d(data["pixel_flags"]):
                        n_spectra[idx] += 1
                        if ref[idx] is None:
                            ref[idx] = mask.copy()
                            ref_file[idx] = path
                        elif not np.array_equal(mask, ref[idx]):
                            n_pix = np.sum(mask != ref[idx])
                            print(f"DIFF  hdus[{idx}] {path}: {n_pix}/{mask.size} pixels differ from {ref_file[idx]}")
                            n_diff[idx] += 1
        except Exception as e:
            print(f"SKIP  {path}: {e}")
            n_skip += 1

    print()
    for idx in (3, 4):
        if n_spectra[idx] == 0:
            print(f"hdus[{idx}]: no spectra found")
        elif n_diff[idx] == 0:
            print(f"hdus[{idx}]: all {n_spectra[idx]} spectra share one mask "
                  f"({np.sum(ref[idx] != 0)}/{ref[idx].size} pixels flagged)")
        else:
            print(f"hdus[{idx}]: {n_diff[idx]}/{n_spectra[idx]} spectra differ from reference ({ref_file[idx]})")
    if n_skip:
        print(f"{n_skip} files skipped")


if __name__ == "__main__":
    main()
