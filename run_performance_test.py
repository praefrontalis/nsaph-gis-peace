import logging
import time

import colorlog
import matplotlib.pyplot as plt
from aggregator import aggregate, parse_options

colors = {"AVG_MONTE_CARLO": "blue", "AVG_AREA_WEIGHTED": "orange"}


def measure_time(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


def run_performance_test(
    geo_shp,
    geo_id,
    geo_add,
    obs_shp,
    obs_ids,
    time_agg,
    time_start_id,
    time_end_id,
    time_id_format,
    epsg):

    method_ns = {
        "AVG_AREA_WEIGHTED": [0],
        "AVG_MONTE_CARLO": [10, 20, 30],  # [100, 200, 500],
    }

    fig, ax = plt.subplots()

    for method, ns in method_ns.items():
        for i, n in enumerate(ns):
            time_elapsed = measure_time(
                    aggregate,
                    geo_shp,
                    geo_id,
                    geo_add,
                    obs_shp,
                    obs_ids,
                    method,
                    time_agg,
                    time_start_id,
                    time_end_id,
                    time_id_format,
                    epsg,
                    n,
                    max_error=None
                )
            if not i:
                ax.plot(n, time_elapsed, "o", color=colors[method], label=method)
            else:
                ax.plot(n, time_elapsed, "o", color=colors[method])
    ax.legend()

    # plt.show()
    plt.savefig("performance_test.png")


if __name__ == "__main__":
    opts = parse_options()

    # logging settings
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(message)s'))
    level = logging.getLevelName(opts.log_level)
    logging.basicConfig(
        level=level,
        handlers=[handler])
    logger = logging.getLogger()

    run_performance_test(
        opts.geo_shp,
        opts.geo_id,
        opts.geo_add,
        opts.obs_shp,
        opts.obs_ids,
        opts.time_agg,
        opts.time_start_id,
        opts.time_end_id,
        opts.time_id_format,
        opts.epsg
    )
