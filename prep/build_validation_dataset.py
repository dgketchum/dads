import os
import multiprocessing
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
from tqdm import tqdm
from cube.model.value_limits import TARGET_LIMITS


def _unique_val_stations(val_dir: str, limit_files: int = 0) -> List[str]:
    files = [
        os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".parquet")
    ]
    files.sort()
    if limit_files and limit_files > 0:
        files = files[:limit_files]
    stations = []
    for p in files:
        try:
            col = pd.read_parquet(p, columns=["station"])
            stations.extend(col["station"].astype(str).unique().tolist())
        except Exception:
            pass
    stations = sorted(set(stations))
    return stations


def _load_obs_series(
    train_parquet_root: str, target_var: str, station: str
) -> Optional[pd.Series]:
    p = os.path.join(train_parquet_root, target_var, f"{station}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    if target_var not in df.columns:
        return None
    s = df[target_var].astype(np.float32)
    s.index = pd.to_datetime(df.index)
    return s


def _get_series(gridded_root: str, station: str, var_name: str) -> Optional[pd.Series]:
    p = os.path.join(gridded_root, f"{station}.parquet")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_parquet(p)
    except ArrowInvalid:
        return None
    s = pd.Series(df[var_name].astype(np.float32).values, index=df.index)
    s = s.sort_index()
    if "gridmet" in gridded_root and var_name in ["tmmx", "tmmn"]:
        s = s - 273.15
    if "era5_land" in gridded_root:
        pass  # era5_land only has from 2002
    return s


def _process_one_station(
    args: Tuple[
        str, str, str, Dict[str, str], Dict[str, str], str, Dict[str, str], bool
    ],
):
    (
        st,
        training_parquet_root,
        target_var,
        product_dirs,
        var_map,
        out_dir,
        agg_map,
        overwrite,
    ) = args
    out_path = os.path.join(out_dir, f"{st}.parquet")
    if os.path.exists(out_path) and not overwrite:
        return st
    obs = _load_obs_series(training_parquet_root, target_var, st)
    if obs is None or obs.empty:
        return st
    frames = {"obs": obs}
    for product, pdir in product_dirs.items():
        if product not in var_map:
            continue
        vn = var_map[product]
        s = _get_series(pdir, st, vn)
        if s is None:
            continue
        frames[product] = s
    if len(frames) <= 1:
        return st
    base = frames["obs"]
    base = base.dropna()
    if base.empty:
        return st
    join = pd.DataFrame({"obs": base})
    for product, s in frames.items():
        if product == "obs":
            continue
        s = s.reindex(join.index)
        join[product] = s
    if join.empty:
        return st
    # Apply target limits to obs and products (set out-of-range to NaN)
    base_var = target_var.replace("_obs", "")
    lim = TARGET_LIMITS.get(base_var)
    if lim is not None:
        lo, hi = float(lim[0]), float(lim[1])
        for c in list(join.columns):
            v = pd.to_numeric(join[c], errors="coerce")
            join.loc[(v < lo) | (v > hi), c] = np.nan
    # Drop rows with NaN obs after clamping
    join = join.dropna(subset=["obs"])
    if join.empty:
        return st
    out = pd.DataFrame({"station": st, "dt": join.index.strftime("%Y-%m-%d")})
    out["obs"] = join["obs"].astype(np.float32).values
    for product in var_map.keys():
        if product in join.columns:
            out[product] = join[product].astype(np.float32).values
        else:
            out[product] = np.nan
    out.to_parquet(out_path)
    # print(out.shape, out.columns.to_list())
    return st


def build_validation_dataset(
    lgbm_root: str,
    training_parquet_root: str,
    target_var: str,
    product_dirs: Dict[str, str],
    var_map: Dict[str, str],
    out_dir: str,
    limit_val_files: int = 0,
    agg_map: Optional[Dict[str, str]] = None,
    num_workers: int = 1,
    overwrite: bool = False,
):
    val_dir = os.path.join(lgbm_root, target_var, "val")
    stations = _unique_val_stations(val_dir, limit_files=limit_val_files)
    if not stations:
        return None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if agg_map is None:
        agg_map = {}

    if num_workers is None or num_workers <= 1:
        for st in tqdm(stations, total=len(stations), desc=f"Build val {target_var}"):
            _process_one_station(
                (
                    st,
                    training_parquet_root,
                    target_var,
                    product_dirs,
                    var_map,
                    out_dir,
                    agg_map or {},
                    overwrite,
                )
            )
    else:
        args = [
            (
                st,
                training_parquet_root,
                target_var,
                product_dirs,
                var_map,
                out_dir,
                agg_map or {},
                overwrite,
            )
            for st in stations
        ]
        with multiprocessing.Pool(processes=int(num_workers)) as pool:
            pbar = tqdm(total=len(stations), desc=f"Build val {target_var}")
            for _ in pool.imap_unordered(_process_one_station, args):
                pbar.update(1)
            pbar.close()
    return None


if __name__ == "__main__":
    home = os.path.expanduser("~")
    d = os.path.join("/data", "ssd2", "dads")
    share = os.path.join(
        home, "data", "IrrigationGIS", "dads", "met", "gridded", "processed_parquet"
    )

    lgbm_root_ = os.path.join(d, "training", "lgbm")
    train_parquet_root_ = os.path.join(d, "training", "parquet")

    prism_dir_ = os.path.join(share, "prism")
    gridmet_dir_ = os.path.join(share, "gridmet")
    nldas_dir_ = os.path.join(share, "nldas2")
    era5_dir_ = os.path.join(d, "era5_land", "processed_parquet", "daily")

    # conus404_dir_ = os.path.join(share, 'data', 'conus404', 'station_data')

    products_ = {
        "prism": prism_dir_,
        "gridmet": gridmet_dir_,
        "nldas2": nldas_dir_,
        "era5land": era5_dir_,
        # 'conus404': conus404_dir_,
    }

    var_ = "tmax"
    target_var_ = f"{var_}_obs"
    var_map_ = {
        "nldas2": "tmax",
        "prism": "tmax",
        "gridmet": "tmmx",
        "era5land": "tmax",
        # 'conus404': 't2m',  # still don't have
    }

    out_dir_ = os.path.join(d, "gridded_comparison_data", var_)

    agg_map_ = {"nldas2": "mean", "era5land": "mean", "conus404": "mean"}

    build_validation_dataset(
        lgbm_root_,
        train_parquet_root_,
        target_var_,
        products_,
        var_map_,
        out_dir_,
        limit_val_files=0,
        agg_map=agg_map_,
        num_workers=16,
        overwrite=False,
    )
# ========================= EOF ====================================================================
