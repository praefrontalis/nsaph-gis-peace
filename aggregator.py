import argparse
import logging
from datetime import timedelta, datetime, date

import colorlog
import geopandas as gpd
import pandas as pd


intersection_col = "aggregator_intersection"
geo_area_col = "aggregator_geo_area"
intersection_area_col = "aggregator_intersection_area"
area_weight_col = "aggregator_area_weight"


def aggregate_max(
        joined_df: gpd.GeoDataFrame,
        group_by_ids: list,
        geo_adds: list,
        obs_ids: list
):
    # aggregation functions: max for user defined for aggregation
    aggregation_dict = {obs_id: "max" for obs_id in obs_ids}
    # first for user defined for leaving
    for geo_add in geo_adds:
        aggregation_dict[geo_add] = "first"

    joined_df_aggregated = joined_df.groupby(
        by=group_by_ids, as_index=False, sort=False).agg(func=aggregation_dict)

    return joined_df_aggregated


def aggregate_avg_MC(obs_df, geo_df):
    """
    Using random sampling (Monte-Carlo).
    The idea is to generate points with random coordinates
    (either for the whole area or for each region)
    and random time within a given time period.
    For each point test if it is within any
    of the observational shapes and if yes,
    assign the observational value (e.g. density),
    otherwise assign 0.
    Then calculate mean value (e.g. mean density)
    for all points within a geographic shape.
    :param obs_df:
    :param geo_df:
    :return:
    """
    raise NotImplementedError


def add_intersect_area(joined_df, obs_df, epsg):

    joined_df = joined_df.to_crs(epsg=epsg)
    obs_df = obs_df.to_crs(epsg=epsg)
    obs_df.geometry = obs_df.geometry.buffer(0)

    joined_df_obs_geometry = obs_df.iloc[joined_df["index_right"]]

    joined_df[intersection_col] = \
        joined_df["geometry"].intersection(joined_df_obs_geometry, align=False)

    joined_df[geo_area_col] = joined_df["geometry"].area
    joined_df[intersection_area_col] = joined_df[intersection_col].area

    joined_df[area_weight_col] = \
        joined_df[intersection_area_col] / joined_df[geo_area_col]

    return joined_df


def aggregate_avg_weighted(
        joined_df, group_by_ids, geo_adds, obs_ids, obs_df, epsg,
        time_agg, time_weight_col):
    """
    Create a set of intersections between geographical
    and observational shapes.

    Each element in this set lies completely
    within one geographic region and within at most one observational shape.
    If an element is not in any of the observational shapes,
    assign value 0 to it, otherwise assign the observed value.
    For each geographic region, calculate weighted average
    of values over shapes (intersections) within it,
    using shapes area as weight.
    Optionally, if required, aggregate over time dimension.

    1. group by geo_id and time_id
    2. sum values of obs_ids columns weighted by area (and by time if it must be)
    :return:
    """
    joined_df = add_intersect_area(joined_df, obs_df, epsg)

    aggregation_dict = {}

    for obs_id in obs_ids:
        obs_id_weighted_col = f"aggregator_{obs_id}_weighted"
        joined_df[obs_id_weighted_col] = joined_df[obs_id] * joined_df[area_weight_col]

        if time_agg is not None:
            joined_df[obs_id_weighted_col] *= joined_df[time_weight_col]

        aggregation_dict[obs_id_weighted_col] = "mean"

    for geo_add in geo_adds:
        aggregation_dict[geo_add] = "first"

    joined_df_aggregated = joined_df.groupby(
        by=group_by_ids, as_index=False, sort=False).agg(func=aggregation_dict)

    return joined_df_aggregated


def annotate_days(df, start_time_col, end_time_col, day_col, weight_col):
    """
    Каждому обозреваемому событию сопоставить дни, в которые это событие наблюдалось,
    и доли дней, в течение которых это событие наблюдалось
    :param df:
    :param start_time_col:
    :param end_time_col:
    :param day_col:
    :param weight_col:
    :return:
    """
    day = timedelta(days=1)
    dfs = []

    df[day_col] = 0
    df[weight_col] = 0

    for i, row in df.iterrows():
        days = []
        weights = []

        start_dt = row[start_time_col]
        end_dt = row[end_time_col]

        if start_dt.date() == end_dt.date():
            df.loc[i, day_col] = start_dt.date()
            df.loc[i, weight_col] = (end_dt - start_dt).total_seconds() / day.total_seconds()
        elif start_dt.date() < end_dt.date():
            start_day_end = start_dt.date() + day
            end_day_start = end_dt.date()

            start_day_end_dt = datetime(start_day_end.year, start_day_end.month, start_day_end.day)
            end_day_start_dt = datetime(end_day_start.year, end_day_start.month, end_day_start.day)

            df.loc[i, day_col] = start_dt.date()
            df.loc[i, weight_col] = (start_day_end_dt - start_dt).total_seconds() / day.total_seconds()

            current_day_end = start_day_end

            while current_day_end < end_day_start:
                days.append(current_day_end)
                weights.append(1)
                current_day_end += day

            days.append(end_day_start)
            weights.append((end_dt - end_day_start_dt).total_seconds() / day.total_seconds())

            add_rows = pd.DataFrame([row]*len(days))
            add_rows[day_col] = days
            add_rows[weight_col] = weights

            dfs.append(add_rows)

        else:
            raise Exception(
                f"Start dt '{start_dt.date()}' "
                f"is larger then end dt '{end_dt.date()}' "
                f"in row '{row}'")

    dfs.append(df)
    df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=df.crs)

    return df


def get_next_month(start_dt):
    is_last_month = not (start_dt.month % 12)
    next_month_year = start_dt.year + 1 if is_last_month else start_dt.year
    next_month = 1 if is_last_month else start_dt.month + 1
    next_month_dt = date(next_month_year, next_month, 1)
    return next_month_dt


def annotate_months(df, start_time_col, end_time_col, month_col, weight_col):
    """
    Каждому обозреваемому событию сопоставить месяцы, в которые это событие наблюдалось,
    и доли месяцев, в течение которых это событие наблюдалось
    :param df:
    :param start_time_col:
    :param end_time_col:
    :param month_col:
    :param weight_col:
    :return:
    """
    dfs = []

    df[month_col] = 0
    df[weight_col] = 0

    for i, row in df.iterrows():
        months = []
        weights = []

        start_dt = row[start_time_col]
        end_dt = row[end_time_col]

        start_month = date(start_dt.year, start_dt.month, 1)
        end_month = date(end_dt.year, end_dt.month, 1)

        start_month_next = get_next_month(start_dt)

        if start_month == end_month:
            df.loc[i, month_col] = start_month
            df.loc[i, weight_col] = (end_dt - start_dt).total_seconds() / (start_month_next - start_month).total_seconds()

        elif start_month < end_month:

            df.loc[i, month_col] = start_month
            df.loc[i, weight_col] = (start_month_next - start_dt).total_seconds() / (start_month_next - start_month).total_seconds()

            current_month = start_month_next

            while start_month_next < end_month:
                months.append(current_month)
                weights.append(1)
                current_month = get_next_month(current_month)

            months.append(start_month_next)
            end_month_next = get_next_month(end_month)
            weights.append((end_dt - end_month).total_seconds() / (end_month_next - end_month).total_seconds())

            add_rows = pd.DataFrame([row] * len(months))
            add_rows[month_col] = months
            add_rows[weight_col] = weights

            dfs.append(add_rows)

        else:
            raise Exception(
                f"Start dt '{start_dt.date()}' "
                f"is larger then end dt '{end_dt.date()}' "
                f"in row '{row}'")

    dfs.append(df)
    df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=df.crs)

    return df


def annotate_years(df, start_time_col, end_time_col, year_col, weight_col):
    """
    Каждому обозреваемому событию сопоставить года, в которые это событие наблюдалось,
    и доли лет, в течение которых это событие наблюдалось
    :param df:
    :param start_time_col:
    :param end_time_col:
    :param year_col:
    :param weight_col:
    :return:
    """
    dfs = []

    df[year_col] = 0
    df[weight_col] = 0

    for i, row in df.iterrows():
        years = []
        weights = []

        start_dt = row[start_time_col]
        end_dt = row[end_time_col]

        start_year = date(start_dt.year, 1, 1)
        end_year = date(end_dt.year, 1, 1)
        start_year_next = date(end_dt.year + 1, 1, 1)

        if start_year == end_year:
            df.loc[i, year_col] = start_year
            df.loc[i, weight_col] = (end_dt - start_dt).total_seconds() / (start_year_next - start_year).total_seconds()

        elif start_year < end_year:

            df.loc[i, year_col] = start_year
            df.loc[i, weight_col] = (start_year_next - start_dt).total_seconds() / (start_year_next - start_year).total_seconds()

            current_year = start_year_next

            while current_year < end_year:
                years.append(current_year)
                weights.append(1)
                current_year = date(current_year.year + 1, 1, 1)

            years.append(current_year)
            end_year_next = date(end_year.year + 1, 1, 1)
            weights.append((end_dt - end_year).total_seconds() / (end_year_next - end_year).total_seconds())

            add_rows = pd.DataFrame([row] * len(years))
            add_rows[year_col] = years
            add_rows[weight_col] = weights

            dfs.append(add_rows)

        else:
            raise Exception(
                f"Start dt '{start_dt.date()}' "
                f"is larger then end dt '{end_dt.date()}' "
                f"in row '{row}'")

    dfs.append(df)
    df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=df.crs)

    return df


def fill_with_0(aggregated_df, geo_df, geo_id):
    aggregated_df_columns = set(aggregated_df.columns)
    geo_df_columns = set(geo_df.columns)

    # which geo locations were lost during aggregation
    geo_df_missing_rows = geo_df.loc[~geo_df[geo_id].isin(aggregated_df[geo_id])].copy()

    # assign 0 to them
    aggregated_df_columns_unique = aggregated_df_columns.difference(geo_df_columns)

    # delete them
    aggregated_df_columns_missing = geo_df_columns.difference(aggregated_df_columns)

    geo_df_missing_rows.drop(columns=aggregated_df_columns_missing, inplace=True)

    geo_df_missing_rows[list(aggregated_df_columns_unique)] = 0

    aggregated_df = aggregated_df.append(geo_df_missing_rows)

    return aggregated_df


def aggregate(
        geo_shp_fp: str,
        geo_id: str,
        geo_adds: list,
        obs_shp_fp: str,
        obs_id: list,
        agg_type: str,
        time_agg: str,
        time_start_id: str,
        time_end_id: str,
        time_id_format: str,
        epsg: int
):
    # df2 = df.drop(labels=range(101, 33144), axis=0)
    # df2.to_file("cb_2017_us_zcta510_100.shp")

    # epsg = 4269

    geo_df = gpd.read_file(geo_shp_fp)  #.set_crs(obs_df.crs)

    if (geo_df[geo_id].value_counts() > 1).any():
        non_unique_values = list((geo_df[geo_id].value_counts() > 1).index)
        logger.warning(
            f"Non unique id column of the geographic region: {geo_id}"
            f"Non unique values: {non_unique_values}")

    obs_df = gpd.read_file(obs_shp_fp).set_crs(geo_df.crs, allow_override=True)

    joined_df = gpd.sjoin(geo_df, obs_df, how="inner", predicate="intersects")

    weight_col = "aggregator_datetime_weight"
    time_period_col = "aggregator_datetime"

    if time_agg is None:
        group_by_ids = [geo_id, ]
    else:
        start_time_col = f"{time_start_id}_datetime"
        end_time_col = f"{time_end_id}_datetime"
        joined_df[start_time_col] = pd.to_datetime(joined_df[time_start_id], format=time_id_format)
        joined_df[end_time_col] = pd.to_datetime(joined_df[time_end_id], format=time_id_format)

        if time_agg == "DAILY":
            joined_df = annotate_days(
                joined_df,
                start_time_col, end_time_col,
                time_period_col,
                weight_col
            )
        elif time_agg == "MONTHLY":
            joined_df = annotate_months(
                joined_df,
                start_time_col, end_time_col,
                time_period_col,
                weight_col)

        elif time_agg == "ANNUAL":
            joined_df = annotate_years(
                joined_df,
                start_time_col, end_time_col,
                time_period_col,
                weight_col)
        else:
            raise NotImplementedError(
                f"Not implemented time aggregation type '{time_agg}'")

        group_by_ids = [geo_id, time_period_col]

    if agg_type == "MAX":
        aggregated_df = aggregate_max(
            joined_df, group_by_ids, geo_adds, obs_id)

    # elif agg_type == "AVG_MONTE_CARLO":
    #     return aggregate_avg_MC(joined_df)

    elif agg_type == "AVG_AREA_WEIGHTED":
        aggregated_df = aggregate_avg_weighted(
            joined_df, group_by_ids, geo_adds, obs_id, obs_df, epsg,
            time_agg, weight_col
        )

    else:
        raise NotImplementedError(
            f"Not implemented type of aggregation: '{agg_type}', "
            f"implement it yourself")

    # assign 0 to areas without observations
    full_df = fill_with_0(aggregated_df, geo_df, geo_id)
    return full_df


def write_shp(
        shp_df: gpd.GeoDataFrame,
        out_shp: str):
    shp_df.to_file(out_shp)


def write_result(
        shp_df: pd.DataFrame,
        out_csv: str):
    shp_df.to_csv(out_csv, index=False)


def parse_options():
    """
    Proposed API

    A function should take the following arguments:

    - A set of geographic shapes, in the form
        of dataframe or shapefile, or an identifier
        (assuming that geographical shapes are loaded
        once in advance of calculations).
        If the input is dataframe, we will also need CRS
        (can be read from shapefile).

    - The name of the identifier column of the geographic region

    - A set of additional columns (fields, annotations)
        from geographic shapes to include in the result

    - A set of observational shapes, in the form of dataframe or shapefile.

    - The name of the value column for observational shapes

    - Aggregation type: MAX, AVG_MONTE_CARLO, AVG_AREA_WEIGHTED

    - If time aggregation is required

    What to return: a list of rows, dataframe or stream of rows

    It should return rows with selected columns
        from the geographic shapes and an additional column
        with aggregated value.
    """
    parser = argparse.ArgumentParser(
        description=
        'Spatial data aggregation. '
        'Attention! '
        'In this code some technical columns are added to datasets '
        'with names starting from "aggregator_". '
        'Do not use such column names in input files to avoid interference!')

    # =================================================================
    # Geographical shape file
    geo_group = parser.add_argument_group('Geographical shape file')

    geo_group.add_argument(
        '--geo-shp',
        help="Path to geographical .shp file",
        type=str
    )

    # TODO: if geometry?
    geo_group.add_argument(
        '--geo-id',
        help="The name of the identifier column of the geographic region",
        type=str
    )

    geo_group.add_argument(
        '--geo-add',
        help="A set of additional columns (fields, annotations) from geographic shapes to include in the result",
        nargs="+",
        type=str
    )

    # =================================================================
    # Observational shape file
    obs_group = parser.add_argument_group('Observational shape file')

    obs_group.add_argument(
        '--obs-shp',
        help="Path to observational .shp file",
        type=str
    )

    obs_group.add_argument(
        '--obs-id',
        help="The name(s) of the value column for observational shapes",
        nargs="+",
        type=str
    )
    # =================================================================
    # Aggregation parameters
    agg_group = parser.add_argument_group('Aggregation parameters')

    agg_group.add_argument(
        '--agg-type',
        help="Aggregation type: MAX|AVG_AREA_WEIGHTED",
        type=str
    )

    agg_group.add_argument(
        '--epsg',
        help="Which EPSG to use when counting area",
        type=int,
        default=5070
    )

    # =================================================================
    # Time aggregation
    time_group = parser.add_argument_group('Time aggregation parameters')

    time_group.add_argument(
        '--time-agg',
        help="Time aggregation type if required: DAILY|MONTHLY|ANNUAL",
        type=str
    )

    time_group.add_argument(
        '--time-start-id',
        help="The name of the datetime column of observation start (for time aggregation)",
        type=str,
        default="Start"
    )

    time_group.add_argument(
        '--time-end-id',
        help="The name of the datetime column of observation end (for time aggregation)",
        type=str,
        default="End"
    )

    time_group.add_argument(
        '--time-id-format',
        help="The datetime format string of columns 'time-start-id' and 'time-end-id' (to parse with datetime.strptime)",
        type=str,
        default="%Y%j %H%M"
    )

    # =================================================================
    # Output options
    out_group = parser.add_argument_group('Output options')

    out_group.add_argument(
        '--out-csv',
        help="Path to output aggregated data .csv",
        type=str
    )

    options = parser.parse_args()

    return options


def main(
    geo_shp,
    geo_id,
    geo_add,
    obs_shp,
    obs_id,
    agg_type,
    time_agg,
    time_start_id,
    time_end_id,
    time_id_format,
    epsg,
    out_csv
):
    # aggregate
    aggregated_df = aggregate(
        geo_shp,
        geo_id,
        geo_add,
        obs_shp,
        obs_id,
        agg_type,
        time_agg,
        time_start_id,
        time_end_id,
        time_id_format,
        epsg
    )

    # write out
    write_result(aggregated_df, out_csv)
    
    
if __name__ == "__main__":
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler])

    logger = logging.getLogger()

    opts = parse_options()

    main(
        opts.geo_shp,
        opts.geo_id,
        opts.geo_add,
        opts.obs_shp,
        opts.obs_id,
        opts.agg_type,
        opts.time_agg,
        opts.time_start_id,
        opts.time_end_id,
        opts.time_id_format,
        opts.epsg,
        opts.out_csv
    )


"""
obs_shp_fp = "/mnt/c/Users/kseniya.petrova/projs/nsaph-gis-data/hms_smoke20220401/hms_smoke20220401.shp"
geo_shp_fp = "/mnt/c/Users/kseniya.petrova/projs/nsaph-gis-data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
"""