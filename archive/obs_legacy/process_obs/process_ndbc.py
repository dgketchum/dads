import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder


def process_single_station(
    station_id,
    ndbc_src,
    ndbc_dst,
    stations_meta,
    overwrite=False,
    min_obs=4,
    debug=False,
):
    out_fp = os.path.join(ndbc_dst, f"{station_id}.parquet")
    if os.path.exists(out_fp) and not overwrite:
        return

    in_fp = os.path.join(ndbc_src, f"{station_id}.parquet")
    if not os.path.exists(in_fp):
        return

    df = pd.read_parquet(in_fp)
    if df.empty:
        return

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # basic per-file observation stats
    try:
        counts_ = df.groupby(df.index.date).size()
        days_ = int(counts_.size)
        obs_total_ = int(len(df))
        min_pd_ = int(counts_.min()) if days_ else 0
        max_pd_ = int(counts_.max()) if days_ else 0
        mean_pd_ = float(counts_.mean()) if days_ else float("nan")
        start_ = pd.to_datetime(df.index.min()).isoformat()
        end_ = pd.to_datetime(df.index.max()).isoformat()
        print(
            f"NDBC {station_id}: {in_fp} rows={obs_total_} days={days_} "
            f"per_day[min/mean/max]={min_pd_}/{mean_pd_:.2f}/{max_pd_} range={start_}→{end_}"
        )
    except Exception:
        pass

    row = stations_meta.loc[station_id]
    lat = float(row["latitude"])
    lon = float(row["longitude"])

    tf = TimezoneFinder()
    tzname = tf.timezone_at(lng=lon, lat=lat)
    local_tz = pytz.timezone(tzname)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(local_tz)

    # ensure numeric before vector components
    if "wind_dir" in df.columns:
        df["wind_dir"] = pd.to_numeric(df["wind_dir"], errors="coerce")
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    if "wind_speed" in df.columns and "wind_dir" in df.columns:
        th = np.deg2rad(df["wind_dir"])
        df["u"] = df["wind_speed"] * (-np.sin(th))
        df["v"] = df["wind_speed"] * (-np.cos(th))

    # coerce other numeric fields to avoid object dtypes in groupby means
    num_cols = [
        "air_temp",
        "dewpoint",
        "pressure",
        "water_temp",
        "wave_height",
        "dominant_wave_period",
        "average_wave_period",
        "mean_wave_dir",
        "visibility",
        "tide",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = df.index.date
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour
    df["is_morning"] = df["hour"].between(0, 8)
    df["is_afternoon"] = df["hour"].between(12, 18)
    grp = df.groupby("date")
    flags = grp[["is_morning", "is_afternoon"]].any()

    daily = {}
    if "air_temp" in df.columns:
        daily["tmax"] = grp["air_temp"].max()
        daily["tmin"] = grp["air_temp"].min()
        daily["tmean"] = grp["air_temp"].mean()

    for col, out in [
        ("dewpoint", "dewpoint"),
        ("pressure", "pressure"),
        ("wind_speed", "wind"),
        ("wind_dir", "wind_dir"),
        ("u", "u"),
        ("v", "v"),
        ("water_temp", "water_temp"),
        ("wave_height", "wave_height"),
        ("dominant_wave_period", "dominant_wave_period"),
        ("average_wave_period", "average_wave_period"),
        ("mean_wave_dir", "mean_wave_dir"),
        ("visibility", "visibility"),
        ("tide", "tide"),
    ]:
        if col in df.columns:
            daily[out] = grp[col].mean()

    daily["doy"] = grp["doy"].first()
    daily["obs_ct"] = grp.size()

    daily_df = pd.DataFrame(daily)
    daily_df = daily_df[daily_df["obs_ct"] >= int(min_obs)].copy()
    daily_df.drop(columns=["obs_ct"], inplace=True)

    idx = pd.to_datetime(daily_df.index)
    daily_df.index = idx.tz_localize(local_tz)
    flags.index = pd.to_datetime(flags.index).tz_localize(local_tz)
    cov = flags.reindex(daily_df.index).fillna(False)
    if "tmin" in daily_df.columns:
        daily_df.loc[~cov["is_morning"].values, "tmin"] = np.nan
    if "tmax" in daily_df.columns:
        daily_df.loc[~cov["is_afternoon"].values, "tmax"] = np.nan
    daily_df["latitude"] = lat
    daily_df["longitude"] = lon

    if daily_df.empty:
        # likely error: daily_df empty after filtering
        try:
            cols_ = list(df.columns)
            print(
                f"NDBC {station_id}: empty daily after filtering. df rows={len(df)} cols={len(cols_)} "
                f"first_date={df.index.min()} last_date={df.index.max()} cols={cols_}"
            )
        except Exception:
            pass
        return

    daily_df.to_parquet(out_fp)
    return


def process_ndbc_daily(
    ndbc_src,
    ndbc_dst,
    stations_meta_csv,
    station_ids=None,
    overwrite=False,
    min_obs=4,
    num_workers=12,
    debug=False,
):
    os.makedirs(ndbc_dst, exist_ok=True)
    meta = pd.read_csv(stations_meta_csv)
    meta.set_index("station_id", inplace=True)

    if station_ids is None:
        files = [f for f in os.listdir(ndbc_src) if f.endswith(".parquet")]
        station_ids = [os.path.splitext(f)[0] for f in files]

    if debug or int(num_workers) == 1:
        for sid in station_ids:
            process_single_station(
                sid,
                ndbc_src,
                ndbc_dst,
                meta,
                overwrite=overwrite,
                min_obs=min_obs,
                debug=debug,
            )
    else:
        args_iter = [
            (sid, ndbc_src, ndbc_dst, meta, overwrite, min_obs, debug)
            for sid in station_ids
        ]
        with Pool(processes=int(num_workers)) as pool:
            for _ in pool.starmap(process_single_station, args_iter):
                pass


def write_ndbc_hourly(ndbc_src, out_root, stations_meta_csv, station_ids=None):
    meta = pd.read_csv(stations_meta_csv)
    meta.set_index("station_id", inplace=True)

    if station_ids is None:
        files = [f for f in os.listdir(ndbc_src) if f.endswith(".parquet")]
        station_ids = [os.path.splitext(f)[0] for f in files]

    for sid in station_ids:
        in_fp = os.path.join(ndbc_src, f"{sid}.parquet")
        if not os.path.exists(in_fp):
            continue

        df = pd.read_parquet(in_fp)
        if df.empty:
            continue

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

        row = meta.loc[sid]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # Map to MADIS-like column names
        colmap = {
            "air_temp": "temperature",
            "dewpoint": "dewpoint",
            "wind_speed": "windSpeed",
            "wind_dir": "windDir",
            "wind_gust": "windGust",
        }
        keep_extra = [
            "pressure",
            "water_temp",
            "wave_height",
            "dominant_wave_period",
            "average_wave_period",
            "mean_wave_dir",
            "visibility",
            "tide",
        ]

        df_m = df.copy()
        for s, t in colmap.items():
            if s in df_m.columns:
                df_m.rename(columns={s: t}, inplace=True)

        cols = ["temperature", "dewpoint", "windSpeed", "windDir", "windGust"]
        cols += [c for c in keep_extra if c in df_m.columns]

        df_m["stationId"] = sid
        df_m["latitude"] = lat
        df_m["longitude"] = lon
        if "elevation" not in df_m.columns:
            df_m["elevation"] = np.nan  # likely missing in NDBC meta
        df_m["code1PST"] = ""
        df_m["code2PST"] = ""

        df_m["datetime"] = df_m.index

        final_cols = [
            "datetime",
            "stationId",
            "elevation",
            "latitude",
            "longitude",
            "code1PST",
            "code2PST",
        ]
        final_cols += [c for c in cols if c in df_m.columns]

        ym = df_m.index.to_period("M")
        for p in ym.unique():
            mask = ym == p
            sub = df_m.loc[mask, final_cols]
            out_dir = os.path.join(out_root, sid)
            os.makedirs(out_dir, exist_ok=True)
            out_fp = os.path.join(out_dir, f"{sid}_{p.strftime('%Y%m')}.parquet")
            sub.to_parquet(out_fp)


if __name__ == "__main__":
    d = "/media/research/IrrigationGIS"
    if not os.path.exists(d):
        d = "/nas"

    ndbc = os.path.join(d, "climate", "ndbc")
    ndbc_meta_csv = os.path.join(ndbc, "ndbc_meta", "ndbc_stations.csv")
    ndbc_src_dir = os.path.join(ndbc, "ndbc_records")
    ndbc_daily_dir = os.path.join(ndbc, "ndbc_daily")

    process_ndbc_daily(
        ndbc_src_dir, ndbc_daily_dir, ndbc_meta_csv, overwrite=False, min_obs=4
    )

# ========================= EOF ====================================================================
