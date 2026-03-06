import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import pyarrow

NDBC_VARS = [
    "tmax",
    "tmin",
    "tmean",
    "wind",
    "wind_dir",
    "u",
    "v",
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


def _promote_lowercase_file(directory, base, ext):
    upper = os.path.join(directory, f"{base}.{ext}")
    lower = os.path.join(directory, f"{base.lower()}.{ext}")
    if not os.path.exists(upper) and os.path.exists(lower):
        os.rename(lower, upper)


def _assess_ndbc_file(path):
    stn = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_parquet(path)
    except pyarrow.lib.ArrowInvalid:
        return {"station": stn}
    if df.empty:
        return {"station": stn}

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    row = {"station": stn}
    for var in NDBC_VARS:
        if var in df.columns:
            mask = df[var].notna()
            cnt = int(mask.sum())
            row[var] = cnt
            if cnt > 0:
                first_idx = df.index[mask][0]
                last_idx = df.index[mask][-1]
                s = (
                    None
                    if pd.isna(first_idx)
                    else pd.to_datetime(first_idx).strftime("%Y-%m-%d")
                )
                e = (
                    None
                    if pd.isna(last_idx)
                    else pd.to_datetime(last_idx).strftime("%Y-%m-%d")
                )
                row[f"start_{var}"] = s if s is not None else "NULL"
                row[f"end_{var}"] = e if e is not None else "NULL"
            else:
                row[f"start_{var}"] = "NULL"
                row[f"end_{var}"] = "NULL"
        else:
            row[var] = 0
            row[f"start_{var}"] = "NULL"
            row[f"end_{var}"] = "NULL"

    for c in ["latitude", "longitude", "elevation"]:
        if c in df.columns:
            try:
                row[c] = float(df[c].iloc[0])
            except Exception:
                pass
    return row


def assess_downloaded_ndbc(
    records_dir,
    out_csv=None,
    joined_dir=None,
    training_dir=None,
    landsat_dir=None,
    cdr_dir=None,
    solrad_dir=None,
    terrain_dir=None,
    out_shp=None,
    num_workers=12,
    debug_limit=None,
):
    files = [
        os.path.join(records_dir, f)
        for f in os.listdir(records_dir)
        if f.endswith(".parquet")
    ]
    # One-time fix: promote lowercase filenames in records_dir to uppercase station IDs
    promoted = []
    for p in files:
        b = os.path.splitext(os.path.basename(p))[0]
        u = b.upper()
        if b != u:
            upath = os.path.join(records_dir, f"{u}.parquet")
            if not os.path.exists(upath):
                os.rename(p, upath)
            promoted.append(upath)
        else:
            promoted.append(p)
    files = promoted
    if debug_limit:
        files = files[: int(debug_limit)]

    summaries = []
    if num_workers is None or num_workers <= 1:
        for p in tqdm(files, total=len(files), desc="Assessing NDBC"):
            row = _assess_ndbc_file(p)
            if row:
                stn = row["station"]
                if joined_dir:
                    _promote_lowercase_file(joined_dir, stn, "parquet")
                    row["in_joined"] = int(
                        os.path.exists(os.path.join(joined_dir, f"{stn}.parquet"))
                    )
                if training_dir:
                    in_training = 0
                    try:
                        subs = [
                            s
                            for s in os.listdir(training_dir)
                            if os.path.isdir(os.path.join(training_dir, s))
                        ]
                        for s in subs:
                            _promote_lowercase_file(
                                os.path.join(training_dir, s), stn, "parquet"
                            )
                            pth = os.path.join(training_dir, s, f"{stn}.parquet")
                            if os.path.exists(pth):
                                in_training = 1
                                break
                    except Exception:
                        in_training = 0
                    row["in_training"] = in_training
                if landsat_dir:
                    _promote_lowercase_file(landsat_dir, stn, "csv")
                    row["has_landsat"] = int(
                        os.path.exists(os.path.join(landsat_dir, f"{stn}.csv"))
                    )
                if cdr_dir:
                    _promote_lowercase_file(cdr_dir, stn, "csv")
                    row["has_cdr"] = int(
                        os.path.exists(os.path.join(cdr_dir, f"{stn}.csv"))
                    )
                if solrad_dir:
                    _promote_lowercase_file(solrad_dir, stn, "csv")
                    row["has_rsun"] = int(
                        os.path.exists(os.path.join(solrad_dir, f"{stn}.csv"))
                    )
                if terrain_dir:
                    _promote_lowercase_file(terrain_dir, stn, "csv")
                    row["has_terrain"] = int(
                        os.path.exists(os.path.join(terrain_dir, f"{stn}.csv"))
                    )
                summaries.append(row)
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
            for row in tqdm(
                ex.map(_assess_ndbc_file, files),
                total=len(files),
                desc="Assessing NDBC",
            ):
                if row:
                    stn = row["station"]
                    if joined_dir:
                        _promote_lowercase_file(joined_dir, stn, "parquet")
                        row["in_joined"] = int(
                            os.path.exists(os.path.join(joined_dir, f"{stn}.parquet"))
                        )
                    if training_dir:
                        in_training = 0
                        try:
                            subs = [
                                s
                                for s in os.listdir(training_dir)
                                if os.path.isdir(os.path.join(training_dir, s))
                            ]
                            for s in subs:
                                _promote_lowercase_file(
                                    os.path.join(training_dir, s), stn, "parquet"
                                )
                                pth = os.path.join(training_dir, s, f"{stn}.parquet")
                                if os.path.exists(pth):
                                    in_training = 1
                                    break
                        except Exception:
                            in_training = 0
                        row["in_training"] = in_training
                    if landsat_dir:
                        _promote_lowercase_file(landsat_dir, stn, "csv")
                        row["has_landsat"] = int(
                            os.path.exists(os.path.join(landsat_dir, f"{stn}.csv"))
                        )
                    if cdr_dir:
                        _promote_lowercase_file(cdr_dir, stn, "csv")
                        row["has_cdr"] = int(
                            os.path.exists(os.path.join(cdr_dir, f"{stn}.csv"))
                        )
                    if solrad_dir:
                        _promote_lowercase_file(solrad_dir, stn, "csv")
                        row["has_rsun"] = int(
                            os.path.exists(os.path.join(solrad_dir, f"{stn}.csv"))
                        )
                    if terrain_dir:
                        _promote_lowercase_file(terrain_dir, stn, "csv")
                        row["has_terrain"] = int(
                            os.path.exists(os.path.join(terrain_dir, f"{stn}.csv"))
                        )
                    summaries.append(row)

    if len(summaries) == 0:
        return pd.DataFrame(columns=["station"])

    df = pd.DataFrame(summaries).sort_values("station").reset_index(drop=True)

    start_cols = [f"start_{v}" for v in NDBC_VARS if f"start_{v}" in df.columns]
    end_cols = [f"end_{v}" for v in NDBC_VARS if f"end_{v}" in df.columns]
    if start_cols:
        s_list = []
        for c in start_cols:
            s = pd.to_datetime(df[c].replace("NULL", pd.NaT), errors="coerce")
            s_list.append(s)
        svals = pd.concat(s_list, axis=1)
        smin = svals.min(axis=1)
        df["start_date"] = smin.dt.strftime("%Y-%m-%d")
        df.loc[smin.isna(), "start_date"] = "NULL"
    if end_cols:
        e_list = []
        for c in end_cols:
            e = pd.to_datetime(df[c].replace("NULL", pd.NaT), errors="coerce")
            e_list.append(e)
        evals = pd.concat(e_list, axis=1)
        emax = evals.max(axis=1)
        df["end_date"] = emax.dt.strftime("%Y-%m-%d")
        df.loc[emax.isna(), "end_date"] = "NULL"

    if out_csv:
        df.to_csv(out_csv, index=False)

    in_joined = df.get("in_joined", pd.Series(0, index=df.index)).fillna(0).astype(int)
    in_training = (
        df.get("in_training", pd.Series(0, index=df.index)).fillna(0).astype(int)
    )

    end_dt = pd.to_datetime(df.get("end_date", "NULL"), errors="coerce")
    pre01_mask = end_dt.notna() & (end_dt < pd.Timestamp("2001-01-01"))
    pre01_ct = int(pre01_mask.sum())
    remain = ~pre01_mask
    join_fail_mask = remain & (in_joined == 0)
    join_fail_ct = int(join_fail_mask.sum())
    train_fail_mask = remain & (in_joined == 1) & (in_training == 0)

    def mcount(col):
        s = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)
        return int((train_fail_mask & (s == 0)).sum())

    miss_rsun = mcount("has_rsun")
    miss_landsat = mcount("has_landsat")
    miss_cdr = mcount("has_cdr")
    miss_terrain = mcount("has_terrain")

    print("Not-in-training summary:")
    print(f"- Ended before 2001-01-01: {pre01_ct}")
    print(f"- Missing joined companion series: {join_fail_ct}")
    print("- Missing ancillary among training-step failures:")
    print(
        f"  rsun={miss_rsun}, landsat={miss_landsat}, cdr={miss_cdr}, terrain={miss_terrain}"
    )

    if out_shp and "latitude" in df.columns and "longitude" in df.columns:
        m = df.dropna(subset=["latitude", "longitude"]).copy()
        if not m.empty:
            gdf = gpd.GeoDataFrame(
                m,
                geometry=gpd.points_from_xy(m["longitude"], m["latitude"]),
                crs="EPSG:4326",
            )
            gdf.to_file(out_shp, crs="EPSG:4326", engine="fiona")

    var_cols = [v for v in NDBC_VARS if v in df.columns]
    if var_cols:
        print("Total observations by variable:")
        for c in var_cols:
            tot = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).sum()
            print(f"  {c}: {int(tot):,}")

    return df


if __name__ == "__main__":
    d = "/media/research/IrrigationGIS"
    if not os.path.exists(d):
        d = "/nas"

    ndbc = os.path.join(d, "climate", "ndbc")
    ndbc_daily = os.path.join(ndbc, "ndbc_daily")

    landsat_ = os.path.join(d, "dads", "rs", "landsat", "station_data")
    cdr_ = os.path.join(d, "dads", "rs", "cdr", "joined")
    solrad = os.path.join(d, "dads", "dem", "rsun_stations")
    terrain = os.path.join(d, "dads", "dem", "terrain", "station_data")

    out_csv_ = os.path.join(d, "dads", "met", "stations", "ndbc_station_counts.csv")
    out_shp_ = os.path.join(d, "dads", "met", "stations", "ndbc_station_counts.shp")

    training = "/data/ssd2/dads/training/parquet"
    joined = "/data/ssd2/dads/met/joined"

    debug_limit_ = None

    assess_downloaded_ndbc(
        ndbc_daily,
        out_csv_,
        joined,
        training,
        landsat_,
        cdr_,
        solrad,
        terrain,
        out_shp=out_shp_,
        num_workers=12,
        debug_limit=debug_limit_,
    )

# ========================= EOF ====================================================================
