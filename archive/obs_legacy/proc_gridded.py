import os
import pytz

import numpy as np
import pandas as pd
from pandarallel import pandarallel
import pyarrow

from refet import calcs

from utils.station_parameters import station_par_map
from utils.calc_eto import calc_asce_params
from timezonefinder import TimezoneFinder

PACIFIC = pytz.timezone("US/Pacific")

NLDAS_RESAMPLE_MAP = {
    "rsds": "sum",
    "rlds": "sum",
    "prcp": "sum",
    "q": "mean",
    "tmin": "min",
    "tmax": "max",
    "tmean": "mean",
    "wind": "mean",
    "ea": "mean",
}

ERA5LAND_RESAMPLE_MAP = {
    "rsds_hourly_mj": "sum",
    "prcp": "sum",
    "tmin": "min",
    "tmax": "max",
    "tmean": "mean",
    "wind": "mean",
    "ea": "mean",
}


def process_gridded_data(
    stations,
    gridded_dir,
    overwrite=False,
    station_type="openet",
    shuffle=True,
    bounds=None,
    hourly=False,
    pd_parallel=False,
    **kwargs,
):
    """"""
    kw = station_par_map(station_type)

    station_list = pd.read_csv(stations, index_col=kw["index"])

    targets = kwargs["targets"]
    if "alt_dirs" in kwargs:
        alt_dirs = kwargs["alt_dirs"]
    else:
        alt_dirs = None

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[
            (station_list[kw["lat"]] < n) & (station_list[kw["lat"]] >= s)
        ]
        station_list = station_list[
            (station_list[kw["lon"]] < e) & (station_list[kw["lon"]] >= w)
        ]

    out_file_ = None
    out_file_gm = None
    out_file_prism = None
    out_file_dm = None
    out_file_era5land = None

    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
        lon, lat, elv = row[kw["lon"]], row[kw["lat"]], row[kw["elev"]]
        if np.isnan(elv):
            print("{} has nan elevation".format(fid))
            continue

        if targets["nldas2"]:
            in_file_ = os.path.join(
                gridded_dir, "raw_parquet", "nldas2", "{}.parquet.gzip".format(fid)
            )

            if not os.path.exists(in_file_):
                print(
                    f"NLDAS at {os.path.basename(in_file_)} does not exist, skipping, {lat:.1f} {lon:.1f}"
                )

            else:
                if hourly:
                    out_file_ = os.path.join(
                        gridded_dir,
                        "processed_parquet",
                        "nldas2_hourly",
                        "{}.parquet".format(fid),
                    )
                else:
                    out_file_ = os.path.join(
                        gridded_dir,
                        "processed_parquet",
                        "nldas2",
                        "{}.parquet".format(fid),
                    )

                if not os.path.exists(out_file_) or overwrite:
                    proc_nldas(
                        in_file_=in_file_,
                        lat=lat,
                        elev=elv,
                        out_file_=out_file_,
                        hourly_=hourly,
                        parallel=pd_parallel,
                    )
                    print("nldas", fid)
                else:
                    print("nldas {} exists".format(fid))
                    pass

        if targets["era5_land"]:
            if alt_dirs:
                in_file_ = os.path.join(
                    alt_dirs["era5_land"],
                    "processed_parquet",
                    "hourly",
                    "{}.parquet.gzip".format(fid),
                )
            else:
                in_file_ = os.path.join(
                    gridded_dir,
                    "raw_parquet",
                    "era5_land",
                    "{}.parquet.gzip".format(fid),
                )

            if not os.path.exists(in_file_):
                print(
                    f"ERA5LAND at {os.path.basename(in_file_)} does not exist, skipping, {lat:.1f} {lon:.1f}"
                )

            else:
                if alt_dirs:
                    out_file_era5land = os.path.join(
                        alt_dirs["era5_land"],
                        "processed_parquet",
                        "daily",
                        "{}.parquet".format(fid),
                    )
                else:
                    out_file_era5land = os.path.join(
                        gridded_dir,
                        "processed_parquet",
                        "era5_land_hourly",
                        "{}.parquet".format(fid),
                    )

                if not os.path.exists(out_file_era5land) or overwrite:
                    proc_era5_land(
                        in_file_=in_file_,
                        lat=lat,
                        lon=lon,
                        elev=elv,
                        out_file_=out_file_era5land,
                        hourly_=hourly,
                        parallel=pd_parallel,
                    )
                    print("era5_land", fid)
                else:
                    print("era5_land {} exists".format(fid))
                    pass

        if targets["gridmet"]:
            in_file_ = os.path.join(
                gridded_dir, "raw_parquet", "gridmet", "{}.parquet.gzip".format(fid)
            )
            if not os.path.exists(in_file_):
                print(
                    f"GridMET at {os.path.basename(in_file_)} does not exist, skipping, {lat:.1f} {lon:.1f}"
                )

            else:
                out_file_gm = os.path.join(
                    gridded_dir,
                    "processed_parquet",
                    "gridmet",
                    "{}.parquet".format(fid),
                )

                if not os.path.exists(out_file_gm) or overwrite:
                    proc_gridmet(
                        in_file_=in_file_,
                        lat=lat,
                        elev=elv,
                        out_file_=out_file_gm,
                        parallel=pd_parallel,
                    )
                    print("gridmet", fid)
                else:
                    pass

        if targets["prism"]:
            in_file_ = os.path.join(
                gridded_dir, "raw_parquet", "prism", "{}.parquet.gzip".format(fid)
            )
            if not os.path.exists(in_file_):
                print(
                    f"PRISM at {os.path.basename(in_file_)} does not exist, skipping, {lat:.1f} {lon:.1f}"
                )

            else:
                out_file_prism = os.path.join(
                    gridded_dir, "processed_parquet", "prism", "{}.parquet".format(fid)
                )

                if not os.path.exists(out_file_prism) or overwrite:
                    proc_prism(in_file_=in_file_, elev=elv, out_file_=out_file_prism)
                    print("prism", fid)
                else:
                    pass

        if targets["daymet"]:
            in_file_ = os.path.join(
                gridded_dir, "raw_parquet", "daymet", "{}.parquet.gzip".format(fid)
            )
            if not os.path.exists(in_file_):
                print(
                    f"DAYMET at {os.path.basename(in_file_)} does not exist, skipping, {lat:.1f} {lon:.1f}"
                )

            else:
                out_file_dm = os.path.join(
                    gridded_dir, "processed_parquet", "daymet", "{}.parquet".format(fid)
                )

                if not os.path.exists(out_file_dm) or overwrite:
                    proc_daymet(in_file_=in_file_, elev=elv, out_file_=out_file_dm)
                    print("daymet", fid)
                else:
                    pass

        # if fid == 'COVM':
        #     target_files = {'nldas2': out_file_, 'gridmet': out_file_gm,
        #                     'prism': out_file_prism, 'daymet': out_file_dm,
        #                     'era5_land': out_file_era5land}
        #
        #     compare_sources(**target_files)


def compare_sources(**targets):
    dfs = {}
    for k, v in targets.items():
        dfs[k] = pd.read_parquet(v).loc["1991-01-01":"2023-12-31"]

    params = ["prcp", "vpd", "tmean", "u2", "eto"]

    for p in params:
        for k, v in dfs.items():
            if k in ["prism", "daymet"] and p in ["u2", "eto"]:
                pass
            else:
                print(f"{p} - {k} mean: {v[p].mean()}")


def proc_era5_land(in_file_, lat, lon, elev, out_file_, hourly_=False, parallel=False):
    try:
        df = pd.read_parquet(in_file_)

        # U is the velocity toward east and V is the velocity toward north
        wind_u = df["wind_u"]
        wind_v = df["wind_v"]
        df["wind"] = np.sqrt(wind_v**2 + wind_u**2)
        df["wind_dir"] = np.degrees(np.arctan2(wind_u, wind_v))

        # shift index back 1 sec to put final value of accumulation period in day of interest
        grouping_key = (df.index - pd.Timedelta("1s")).date
        df["rsds_hourly_joules"] = (
            df.groupby(grouping_key)["ssrd"].diff().fillna(df["ssrd"])
        )
        df["rsds_hourly_mj"] = df["rsds_hourly_joules"] / 1000000

        df["prcp"] = df["precip"]
        df["psurf"] = df["psurf"]

        df["ea"] = 0.61094 * np.exp((17.625 * df["tdew"]) / (df["tdew"] + 243.04))

        df["hour"] = [i.hour for i in df.index]

        df["tmean"] = df["temp"].copy()

        if hourly_:
            df["doy"] = [i.dayofyear for i in df.index]
            df.to_parquet(out_file_)
            return

        df["tmax"] = df["temp"].copy()
        df["tmin"] = df["temp"].copy()

        tf = TimezoneFinder()

        timezone_str = tf.timezone_at(lng=lon, lat=lat)
        if timezone_str is None:
            print(
                f"Could not determine timezone for {os.path.basename(in_file_)} coords ({lat}, {lon})"
            )

        local_tz = pytz.timezone(timezone_str)

        if df.index.tz is None:
            df.index = df.index.tz_localize("GMT")

        df.index = df.index.tz_convert(local_tz)

        df = df.resample("D").agg(ERA5LAND_RESAMPLE_MAP)

        df["rsds"] = df["rsds_hourly_mj"]

        df["doy"] = [i.dayofyear for i in df.index]

        if parallel:
            asce_params = df.parallel_apply(
                calc_asce_params, lat=lat, elev=elev, zw=10, axis=1
            )
        else:
            asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)

        df[["tmean", "vpd", "rn", "u2", "eto"]] = pd.DataFrame(
            asce_params.tolist(), index=df.index
        )

        df["year"] = [i.year for i in df.index]
        df["date_str"] = [i.strftime("%Y-%m-%d") for i in df.index]

    except (KeyError, pyarrow.lib.ArrowInvalid, FileNotFoundError) as exc:
        bad_files = os.path.join(os.path.dirname(__file__), "bad_files.txt")
        with open(bad_files, "a") as f:
            f.write(in_file_ + " " + " ".join(exc.args) + "\n")
        print(os.path.basename(in_file_), " ".join(exc.args))
        return None

    df.to_parquet(out_file_)


def proc_nldas(in_file_, lat, elev, out_file_, hourly_=False, parallel=False):
    # TODO: timezone localizer here
    try:
        df = pd.read_parquet(in_file_)

        wind_u = df["Wind_E"]
        wind_v = df["Wind_N"]
        df["wind"] = np.sqrt(wind_v**2 + wind_u**2)

        df["temp"] = df["Tair"] - 273.15

        df["rsds"] = df["SWdown"] * 0.0036
        df["rlds"] = df["LWdown"] * 0.0036
        df["prcp"] = df["Rainf"]
        df["q"] = df["Qair"]

        df["hour"] = [i.hour for i in df.index]

        df["ea"] = calcs._actual_vapor_pressure(
            pair=calcs._air_pressure(elev), q=df["q"]
        )

        df["tmean"] = df["temp"].copy()

        if hourly_:
            df["doy"] = [i.dayofyear for i in df.index]
            df.to_parquet(out_file_)
            return

        df["tmax"] = df["temp"].copy()
        df["tmin"] = df["temp"].copy()

        df = df.resample("D").agg(NLDAS_RESAMPLE_MAP)

        df["doy"] = [i.dayofyear for i in df.index]

        if parallel:
            asce_params = df.parallel_apply(
                calc_asce_params, lat=lat, elev=elev, zw=10, axis=1
            )
        else:
            asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)

        df[["tmean", "vpd", "rn", "u2", "eto"]] = pd.DataFrame(
            asce_params.tolist(), index=df.index
        )

        df["year"] = [i.year for i in df.index]
        df["date_str"] = [i.strftime("%Y-%m-%d") for i in df.index]

    except (KeyError, pyarrow.lib.ArrowInvalid, FileNotFoundError) as exc:
        bad_files = os.path.join(os.path.dirname(__file__), "bad_files.txt")
        with open(bad_files, "a") as f:
            f.write(in_file_ + " " + " ".join(exc.args) + "\n")
        print(os.path.basename(in_file_), " ".join(exc.args))
        return None

    df.to_parquet(out_file_)


def proc_gridmet(in_file_, lat, elev, out_file_, parallel=False):
    try:
        df = pd.read_parquet(in_file_)

        df["tmin"] = df["tmmn"] - 273.15
        df["tmax"] = df["tmmx"] - 273.15
        df["q"] = df["sph"]

        df.index = pd.to_datetime(df.index)
        df["year"] = [i.year for i in df.index]
        df["doy"] = [i.dayofyear for i in df.index]

        df["ea"] = calcs._actual_vapor_pressure(
            pair=calcs._air_pressure(elev), q=df["q"]
        )

        df["rsds"] = df["srad"] * 0.0864
        df["wind"] = df["vs"]

        df["prcp"] = df["pr"]

        if parallel:
            asce_params = df.parallel_apply(
                calc_asce_params, lat=lat, elev=elev, zw=10, axis=1
            )
        else:
            asce_params = df.apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)

        df[["tmean", "vpd", "rn", "u2", "eto"]] = pd.DataFrame(
            asce_params.tolist(), index=df.index
        )

    except (KeyError, pyarrow.lib.ArrowInvalid, FileNotFoundError) as exc:
        bad_files = os.path.join(os.path.dirname(__file__), "bad_files.txt")
        with open(bad_files, "a") as f:
            f.write(in_file_ + " " + " ".join(exc.args) + "\n")
        print(os.path.basename(in_file_), " ".join(exc.args))
        return None

    df.to_parquet(out_file_)


def proc_prism(in_file_, elev, out_file_):
    try:
        df = pd.read_parquet(in_file_)

        df["vpd"] = (df["vpdmax"] + df["vpdmin"]) * 0.5 * 0.1
        es = 0.6108 * np.exp(17.27 * df["tmean"] / (df["tmean"] + 237.3))
        ea = es - df["vpd"]
        df["q"] = (0.622 * ea) / (calcs._air_pressure(elev) - (0.378 * ea))

        df["prcp"] = df["ppt"]

        df.index = pd.to_datetime(df.index)
        df["year"] = [i.year for i in df.index]
        df["doy"] = [i.dayofyear for i in df.index]

    except (KeyError, pyarrow.lib.ArrowInvalid, FileNotFoundError) as exc:
        bad_files = os.path.join(os.path.dirname(__file__), "bad_files.txt")
        with open(bad_files, "a") as f:
            f.write(in_file_ + " " + " ".join(exc.args) + "\n")
        print(os.path.basename(in_file_), " ".join(exc.args))
        return None

    df.to_parquet(out_file_)


def proc_daymet(in_file_, elev, out_file_):
    try:
        df = pd.read_parquet(in_file_)

        df["tmean"] = (df["tmin"] + df["tmax"]) * 0.5
        es = 0.6108 * np.exp(17.27 * df["tmean"] / (df["tmean"] + 237.3))
        df["vpd"] = df["vp"] / 1000.0
        ea = es - df["vpd"]
        df["q"] = (0.622 * ea) / (calcs._air_pressure(elev) - (0.378 * ea))

        df.index = pd.to_datetime(df.index)
        df["year"] = [i.year for i in df.index]
        df["doy"] = [i.dayofyear for i in df.index]

        df["ea"] = calcs._actual_vapor_pressure(
            pair=calcs._air_pressure(elev), q=df["q"]
        )

        # intentionally verbose
        df["rsds"] = df["srad"] * 0.0864 * df["dayl"] / 86400

    except (KeyError, pyarrow.lib.ArrowInvalid, FileNotFoundError) as exc:
        bad_files = os.path.join(os.path.dirname(__file__), "bad_files.txt")
        with open(bad_files, "a") as f:
            f.write(in_file_ + " " + " ".join(exc.args) + "\n")
        print(os.path.basename(in_file_), " ".join(exc.args))
        return None

    df.to_parquet(out_file_)


if __name__ == "__main__":
    d = "/media/research/IrrigationGIS"
    alt_gridded = "/data/ssd2/dads/era5_land"
    if not os.path.isdir(d):
        home = os.path.expanduser("~")
        d = os.path.join(home, "data", "IrrigationGIS")

    overwrite = False
    processing_targets = {
        "nldas2": False,
        "gridmet": False,
        "prism": False,
        "daymet": False,
        "era5_land": True,
    }

    alt_dirs_ = {"era5_land": alt_gridded}

    args = {"targets": processing_targets, "alt_dirs": alt_dirs_}

    parallel_ = False
    if parallel_:
        pandarallel.initialize(nb_workers=4)

    network = "madis"

    if network == "madis":
        sites = os.path.join(d, "dads", "met", "stations", "madis_17MAY2025_mgrs.csv")

    elif network == "ghcn":
        sites = os.path.join(d, "climate", "stations", "ghcn_stations.csv")

    else:
        raise ValueError

    grid_dirs = os.path.join(d, "dads", "met", "gridded")

    # daymet bounds
    # bounds = (-178., 7., -53., 83.)

    # era5_land download extent
    bounds = (-125, 25, -67, 53)

    process_gridded_data(
        sites,
        grid_dirs,
        overwrite=overwrite,
        station_type=network,
        shuffle=True,
        bounds=bounds,
        hourly=False,
        pd_parallel=parallel_,
        **args,
    )

# ========================= EOF ====================================================================
