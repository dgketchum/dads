import calendar
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cdsapi
import pandas as pd
import xarray as xr

from prep.station_parameters import station_par_map


def download_era5(target_dir, overwrite=False):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_solar_radiation_downwards",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [53, -125, 25, -67],
    }

    client = cdsapi.Client()

    for year in range(2000, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 4:
                continue

            _, num_days = calendar.monthrange(year, month)
            request["day"] = [str(day) for day in range(1, num_days + 1)]
            request["year"] = str(year)
            request["month"] = f"{month:02d}"
            target_nc = os.path.join(target_dir, f"era5_land_{year}_{month:02d}.nc")
            if not overwrite and os.path.exists(target_nc):
                print(f"{os.path.basename(target_nc)} exists, skippping")
                continue
            client.retrieve(dataset, request, target_nc)
            print(f"downloaded {target_nc}")


def _process_nc_file(nc_file, stations, gridded_dir, overwrite, station_type, bounds):
    """"""
    kw = station_par_map(station_type)
    station_list = pd.read_csv(stations, index_col=kw["index"]).sort_index(
        ascending=True
    )

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[
            (station_list[kw["lat"]] < n) & (station_list[kw["lat"]] >= s)
        ]
        station_list = station_list[
            (station_list[kw["lon"]] < e) & (station_list[kw["lon"]] >= w)
        ]
    else:
        ds = xr.open_dataset(nc_file)
        w = float(ds.longitude[0].values)
        e = float(ds.longitude[-1].values)
        s = float(ds.latitude[-1].values)
        n = float(ds.latitude[0].values)
        ds.close()
        station_list = station_list[
            (station_list[kw["lat"]] < n) & (station_list[kw["lat"]] >= s)
        ]
        station_list = station_list[
            (station_list[kw["lon"]] < e) & (station_list[kw["lon"]] >= w)
        ]

    record_ct = station_list.shape[0]

    ds = xr.open_dataset(nc_file)
    splt = os.path.basename(nc_file).strip(".nc").split("_")
    year, month = splt[-2], splt[-1]

    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):
        lon, lat, _elv = row[kw["lon"]], row[kw["lat"]], row[kw["elev"]]
        print("{}: {} of {}; {:.2f}, {:.2f}".format(fid, i, record_ct, lat, lon))

        sub_dir = os.path.join(gridded_dir, fid)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        _file = os.path.join(sub_dir, f"{fid}_{year}_{month}.parquet")
        if not os.path.exists(_file) or overwrite:
            df = ds.sel(longitude=lon, latitude=lat, method="nearest").to_dataframe()
            df = df[
                [
                    "u10",
                    "v10",
                    "d2m",
                    "t2m",
                    "sp",
                    "tp",
                    "ssrd",
                    "latitude",
                    "longitude",
                ]
            ]
            df = df.rename(
                columns={
                    "u10": "wind_u",
                    "v10": "wind_v",
                    "d2m": "tdew",
                    "t2m": "temp",
                    "sp": "psurf",
                    "tp": "precip",
                    "ssr": "rsds",
                }
            )
            df["temp"] -= 273.15
            df["tdew"] -= 273.15
            df["slat"] = lat
            df["slon"] = lon
            df.to_parquet(_file)
            print("Data extracted and saved to", _file)
        else:
            print("{} exists".format(_file))

    ds.close()


def extract_met_data(
    stations,
    gridded_dir,
    nc_dir,
    overwrite=False,
    station_type="dads",
    bounds=None,
    n_workers=1,
):

    nc_files = sorted([f for f in os.listdir(nc_dir) if f[-10:-3] == "2017_02"])
    nc_files = sorted(
        [
            os.path.join(nc_dir, f)
            for f in os.listdir(nc_dir)
            if "era5_land_" in f and f in nc_files
        ]
    )

    if n_workers == 1:
        for nc_file in nc_files:
            _process_nc_file(
                nc_file, stations, gridded_dir, overwrite, station_type, bounds
            )

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for nc_file in nc_files:
            future = executor.submit(
                _process_nc_file,
                nc_file,
                stations,
                gridded_dir,
                overwrite,
                station_type,
                bounds,
            )
            futures.append(future)
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    home = os.path.expanduser("~")

    d = os.path.join("/data/ssd2/dads")
    era5 = os.path.join(d, "era5_land")

    if not os.path.isdir(d):
        d = os.path.join(home, "data", "IrrigationGIS")
        era5 = os.path.join(d, "climate", "era5")

    nc_dir_ = os.path.join(era5, "netCDF")
    out_files = os.path.join(era5, "raw_parquet")

    # download_era5(nc_dir_, overwrite=False)

    dads = os.path.join(home, "data", "IrrigationGIS", "dads")
    climate = os.path.join(home, "data", "IrrigationGIS", "climate")
    if not os.path.exists(dads):
        dads = os.path.join("/media/research", "IrrigationGIS", "dads")
        climate = os.path.join("/media/research", "IrrigationGIS", "climate")

    # sites = os.path.join(climate, 'ghcn', 'stations', 'ghcn_CANUSA_stations_mgrs.csv')
    # stype = 'ghcn'

    sites = os.path.join(dads, "met", "stations", "madis_17MAY2025_mgrs.csv")
    stype = "madis"

    extract_met_data(
        sites,
        out_files,
        nc_dir=nc_dir_,
        n_workers=1,
        bounds=None,
        overwrite=False,
        station_type=stype,
    )

# ========================= EOF ====================================================================
