"""
Подготовить файл со всеми наблюдаемыми облаками
"""
import argparse
import glob
import os

import geopandas as gpd
import pandas as pd


def concat_all_shp(root_dirs):

    shp_files = []

    for root_dir in root_dirs:
        shp_files += glob.glob(os.path.join(root_dir, "**/*.shp"), recursive=True)

    gdfs_list = []
    crs = None

    files = 0

    for shp_file in shp_files:
        try:
            gdf = gpd.read_file(shp_file)
            if crs is None:
                crs = gdf.crs
            elif crs != gdf.crs:
                gdf = gdf.to_crs(crs)

            gdfs_list.append(gdf)

            files += 1

        except Exception as ex:
            print(f"file '{shp_file}' exception: '{ex}'")

    if crs is not None:
        print(f"... {files} files considered")
        all_df = gpd.GeoDataFrame(pd.concat(gdfs_list, ignore_index=True), crs=crs)
        return all_df
    else:
        return None


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root-dirs',
        nargs="+",
        help="Path to folder with geographical .shp files",
        type=str
    )

    parser.add_argument(
        '--out-shp',
        help="Path to result .shp file",
        type=str
    )
    options = parser.parse_args()

    return options


if __name__ == "__main__":

    opts = parse_options()
    big_gdf = concat_all_shp(opts.root_dirs)
    big_gdf.to_file(opts.out_shp)
