"""
Random subsample from geo-shape file
"""
import argparse

import geopandas as gpd
import numpy as np


def get_random(input_fp, sub_fraction, sub_size):

    gdf = gpd.read_file(input_fp)
    print(f"Read {len(gdf)} rows from file '{input_fp}'")
    sub_gdf = gdf.sample(n=sub_size, frac=sub_fraction, axis="index")
    return sub_gdf


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--in-fp',
        help="Path to geographical .shp file",
        type=str
    )

    parser.add_argument(
        '--out-fp',
        help="Path to result .shp file",
        type=str
    )

    parser.add_argument(
        '--random-seed',
        help="Random seed",
        type=int,
        default=42
    )

    sub = parser.add_mutually_exclusive_group(required=True)

    sub.add_argument(
        '--sub-fraction',
        help="Fraction of file to subsample",
        type=float
    )

    sub.add_argument(
        '--sub-size',
        help="Number of rows to subsample",
        type=int
    )
    options = parser.parse_args()

    return options


def main(input_fp, fraction, sub_size):
    shp = get_random(input_fp, fraction, sub_size)

    shp.to_file(opts.out_fp)
    print(f"Wrote {len(shp)} rows to file '{opts.out_fp}'")


if __name__ == "__main__":

    opts = parse_options()
    np.random.seed(opts.random_seed)
    main(opts.in_fp, opts.sub_fraction, opts.sub_size)
