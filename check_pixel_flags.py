"""Verify that the 'pixel_flags' column is identical between hdus[3] and hdus[4]
(the two telescopes) for every FITS file in a directory.

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

    n_same = n_diff = n_skip = 0
    for path in files:
        try:
            with fits.open(path) as hdus:
                a = hdus[3].data["pixel_flags"]
                b = hdus[4].data["pixel_flags"]
        except Exception as e:
            print(f"SKIP  {path}: {e}")
            n_skip += 1
            continue

        if a.shape != b.shape:
            print(f"DIFF  {path}: shape mismatch {a.shape} vs {b.shape}")
            n_diff += 1
        elif np.array_equal(a, b):
            n_same += 1
        else:
            n_mismatch = np.sum(a != b)
            print(f"DIFF  {path}: {n_mismatch}/{a.size} values differ")
            n_diff += 1

    print(f"\n{len(files)} files: {n_same} identical, {n_diff} different, {n_skip} skipped")


if __name__ == "__main__":
    main()
