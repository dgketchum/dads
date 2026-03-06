import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.elevation import elevation_from_coordinate


def process_weather_stations(stations_csv, shapefiles_dir, flag_file, mgrs, out_csv):
    sdf = pd.read_csv(stations_csv, index_col="fid")
    gdf = gpd.GeoDataFrame(
        sdf, geometry=gpd.points_from_xy(sdf.longitude, sdf.latitude)
    )

    sdf["privileged"] = 0

    mgrs_gdf = gpd.read_file(mgrs)

    shapefile_list = [f for f in os.listdir(shapefiles_dir) if f.endswith(".shp")]
    flagged_stations, ct = {}, 0

    for i, shp in enumerate(shapefile_list, start=1):
        print("{} of {}; {}".format(i, len(shapefile_list), os.path.basename(shp)))

        shapefile_path = os.path.join(shapefiles_dir, shp)
        new_stations_gdf = gpd.read_file(shapefile_path)

        for index, row in new_stations_gdf.iterrows():
            existing_station = sdf[sdf.index == row["index"]]

            if not existing_station.empty:
                lat1, lon1 = (
                    existing_station["latitude"].iloc[0],
                    existing_station["longitude"].iloc[0],
                )
                lat2, lon2 = (row["latitude"], row["longitude"])
                distance = haversine_distance(lat1, lon1, lat2, lon2) * 1000

                if distance > 250:
                    if row["index"] in flagged_stations:
                        flagged_stations[row["index"]].append((shp[19:27], distance))
                    else:
                        flagged_stations[row["index"]] = [(shp[19:27], distance)]
            else:
                try:
                    point = Point(row["longitude"], row["latitude"])
                    mgrs_tile = mgrs_gdf[mgrs_gdf.contains(point)]["MGRS"].values[0]

                    elev = elevation_from_coordinate(row["latitude"], row["longitude"])
                    new_record = {
                        "index": row["index"],
                        "name": "None",
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "elevation": elev,
                        "fid": row["index"],
                        "orig_netid": row["index"],
                        "privileged": 1,
                        "MGRS_TILE": mgrs_tile,
                    }

                    sdf.loc[row["index"]] = new_record

                    print(
                        "add station {}; {:.2f}, {:.2f} at {:.2f} m".format(
                            row["index"], row["latitude"], row["longitude"], elev
                        )
                    )
                    ct += 1

                except Exception as e:
                    print(index, e)
                    continue

    with open(flag_file, "w") as f:
        json.dump(flagged_stations, f, indent=4)

    sdf.to_csv(out_csv, index=False)
    geometry = [Point(xy) for xy in zip(sdf["longitude"], sdf["latitude"])]
    gdf = gpd.GeoDataFrame(sdf, geometry=geometry)
    gdf.to_file(out_csv.replace(".csv", ".shp"), crs="EPSG:4326", engine="fiona")
    print(out_csv, ct, "added stations")


def haversine_distance(lat1_, lon1_, lat2_, lon2_):
    R = 6371
    dlat = np.radians(lat2_ - lat1_)
    dlon = np.radians(lon2_ - lon1_)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1_)) * np.cos(
        np.radians(lat2_)
    ) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def get_station_metadata(
    data_directory,
    dataset_source_label,
    csv_dir,
    output_station_tracker_path,
    open_nc_func,
    metadata_fields,
    time_constraint_yearmonths=None,
    input_station_tracker_path=None,
):
    station_dct = {}
    if input_station_tracker_path and os.path.exists(input_station_tracker_path):
        try:
            with open(input_station_tracker_path, "r") as fp:
                content = fp.read()
                if content and content.strip():
                    station_dct = json.loads(content)
        except json.JSONDecodeError:
            station_dct = {}
        except OSError:
            station_dct = {}

    station_to_yearmonths_map = defaultdict(set)
    start_ym_str_constraint = None
    end_ym_str_constraint = None
    if time_constraint_yearmonths:
        if (
            isinstance(time_constraint_yearmonths, (list, tuple))
            and len(time_constraint_yearmonths) == 2
        ):
            start_ym_str_constraint = str(time_constraint_yearmonths[0])
            end_ym_str_constraint = str(time_constraint_yearmonths[1])

    for station_dir_name in os.listdir(csv_dir):
        station_id = station_dir_name
        station_csv_files_path = os.path.join(csv_dir, station_id)
        for csv_file in os.listdir(station_csv_files_path):
            if csv_file.endswith(".csv"):
                parts = csv_file.split("_")
                if len(parts) > 1:
                    yearmonth = parts[1].split(".")[0]
                    if start_ym_str_constraint and end_ym_str_constraint:
                        if not (
                            start_ym_str_constraint
                            <= yearmonth
                            <= end_ym_str_constraint
                        ):
                            continue
                    station_to_yearmonths_map[station_id].add(yearmonth)

    if not station_to_yearmonths_map:
        with open(output_station_tracker_path, "w") as fp:
            json.dump(station_dct, fp, indent=4)
        return

    monthly_targets_remaining = defaultdict(set)
    stations_present_in_csv_dir_fs = set()
    for s_dir_name in os.listdir(csv_dir):
        stations_present_in_csv_dir_fs.add(s_dir_name)

    for station_id_map, yms_set in station_to_yearmonths_map.items():
        if station_id_map in stations_present_in_csv_dir_fs:
            if station_id_map not in station_dct:
                for ym_str_map in yms_set:
                    monthly_targets_remaining[ym_str_map].add(station_id_map)

    if not any(monthly_targets_remaining.values()):
        with open(output_station_tracker_path, "w") as fp:
            json.dump(station_dct, fp, indent=4)
        return

    all_gz_files_in_datadir = glob.glob(os.path.join(data_directory, "*.gz"))

    while any(monthly_targets_remaining.values()):
        active_monthly_targets = {
            ym: stations
            for ym, stations in monthly_targets_remaining.items()
            if stations
        }
        if not active_monthly_targets:
            break

        yearmonth_to_process = max(
            active_monthly_targets, key=lambda k: len(active_monthly_targets[k])
        )
        current_targets_for_this_ym_iteration = set(
            monthly_targets_remaining[yearmonth_to_process]
        )

        if not current_targets_for_this_ym_iteration:
            if yearmonth_to_process in monthly_targets_remaining:
                del monthly_targets_remaining[yearmonth_to_process]
            continue

        stations_processed_in_current_nc_loop = set()

        yearmo_nc_files_for_processing = []
        for f_path in all_gz_files_in_datadir:
            if yearmonth_to_process in os.path.basename(f_path):
                file_size = os.path.getsize(f_path)
                yearmo_nc_files_for_processing.append(
                    {"path": f_path, "size": file_size}
                )

        sorted_yearmo_nc_files = [
            item["path"]
            for item in sorted(
                yearmo_nc_files_for_processing, key=lambda x: x["size"], reverse=True
            )
        ]

        for enum, filename in enumerate(sorted_yearmo_nc_files, start=1):
            ds = open_nc_func(filename)
            if ds is None:
                continue

            required_vars_in_ds = ["stationId"] + metadata_fields
            missing_vars = [var for var in required_vars_in_ds if var not in ds]
            if missing_vars:
                continue

            valid_data = ds[required_vars_in_ds]
            df = valid_data.to_dataframe()

            if "stationId" not in df.columns and "stationId" in df.index.names:
                df = df.reset_index()

            if "stationId" not in df.columns:
                continue

            df["stationId"] = df["stationId"].astype(str)
            df = df.set_index("stationId", drop=False)
            df.dropna(subset=metadata_fields, how="all", inplace=True)

            file_updated_tracker = False
            for station_id_in_file in df.index:
                if station_id_in_file in current_targets_for_this_ym_iteration:
                    stations_processed_in_current_nc_loop.add(station_id_in_file)

                    if station_id_in_file in station_dct:
                        monthly_targets_remaining[yearmonth_to_process].discard(
                            station_id_in_file
                        )
                        continue

                    row = df.loc[station_id_in_file]
                    stype_val = row["stationType"]
                    stype_str = (
                        stype_val.decode("utf-8", errors="replace")
                        if isinstance(stype_val, bytes)
                        else (str(stype_val) if pd.notna(stype_val) else None)
                    )

                    lat_val = row["latitude"]
                    lon_val = row["longitude"]
                    elev_val = row["elevation"]

                    try:
                        current_station_data_from_nc = {
                            "lat": float(lat_val),
                            "lon": float(lon_val),
                            "elev": float(elev_val),
                            "stype": stype_str,
                        }
                    except ValueError:
                        continue

                    station_dct[station_id_in_file] = current_station_data_from_nc
                    station_dct[station_id_in_file]["sources"] = [dataset_source_label]
                    file_updated_tracker = True

                    monthly_targets_remaining[yearmonth_to_process].discard(
                        station_id_in_file
                    )

            if file_updated_tracker:
                print(
                    f"{os.path.basename(filename)}: {enum} of {len(sorted_yearmo_nc_files)}, "
                    f"{len(station_dct)} stations",
                    flush=True,
                )
                with open(output_station_tracker_path, "w") as fp:
                    json.dump(station_dct, fp, indent=4)

            if stations_processed_in_current_nc_loop.issuperset(
                current_targets_for_this_ym_iteration
            ):
                break

        if not monthly_targets_remaining.get(yearmonth_to_process):
            if yearmonth_to_process in monthly_targets_remaining:
                del monthly_targets_remaining[yearmonth_to_process]

    with open(output_station_tracker_path, "w") as fp:
        json.dump(station_dct, fp, indent=4)


def write_stations_to_shapefile(station_tracker, shapefile_path, existing_check=None):
    with open(station_tracker, "r") as f:
        station_dct = json.load(f)
    print(len(station_dct))
    data = []
    if existing_check:
        exists = [s.split(".")[0] for s in os.listdir(existing_check)]
        station_dct = {k: v for k, v in station_dct.items() if k not in exists}

    for station_id, info in station_dct.items():
        geo_ = Point(info["lon"], info["lat"])
        stype = info["stype"]
        entry = {
            "fid": station_id,
            "latitude": info["lat"],
            "longitude": info["lon"],
            "elev": info["elev"],
            "stype": stype,
            "geometry": geo_,
        }
        entry = validate_entry(station_id, entry)

        if entry:
            data.append(entry)

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    print(gdf.shape[0], "shapefile")
    gdf.to_file(shapefile_path)
    df = gdf[[c for c in gdf.columns if c != "geometry"]]
    df.to_csv(shapefile_path.replace(".shp", ".csv"))


def validate_entry(station_id, info):
    try:
        if not all(key in info for key in ("latitude", "longitude", "elev", "stype")):
            raise ValueError("Missing required key in station information")

        lat = float(info["latitude"])
        if not -90 <= lat <= 90:
            raise ValueError("Invalid latitude value")

        lon = float(info["longitude"])
        if not -180 <= lon <= 180:
            raise ValueError("Invalid longitude value")

        elev = float(info["elev"])
        stype = str(info["stype"])
        geo_ = Point(lon, lat)

        entry = {
            "fid": station_id,
            "latitude": lat,
            "longitude": lon,
            "elev": elev,
            "stype": stype,
            "geometry": geo_,
        }

        return entry

    except (ValueError, TypeError) as e:
        print(f"Error validating station {station_id}: {e}")
        return None


def write_missing(madis, dads, missing):
    sdf = gpd.read_file(madis)
    sdf.index = sdf["fid"]

    ddf = gpd.read_file(dads)
    ddf.index = ddf["fid"]
    missing_idx = [i for i in sdf.index if i not in ddf.index]

    sdf = sdf.loc[missing_idx]
    sdf.index = [i for i in range(sdf.shape[0])]
    sdf.to_file(missing)


if __name__ == "__main__":
    d = "/media/research/IrrigationGIS"
    if not os.path.exists(d):
        d = "/nas"

    madis = os.path.join(d, "climate", "madis")
    dads = os.path.join(d, "dads")

    mesonet_dir_research = os.path.join(madis, "LDAD", "mesonet", "netCDF")
    mesonet_csv_research = os.path.join(madis, "LDAD", "mesonet", "inclusive_csv")

    mesonet_dir_public = os.path.join(madis, "LDAD_public", "mesonet", "netCDF")
    mesonet_csv_public = os.path.join(madis, "LDAD_public", "mesonet", "inclusive_csv")

    tracker_old = os.path.join(madis, "madis_meta_15MAY2025.json")
    tracker_new = os.path.join(madis, "madis_meta_17MAY2025.json")

    # shp = os.path.join(dads, 'met', 'stations', 'madis_17MAY2025.shp')

    # get_station_metadata(data_directory=mesonet_dir_research,
    #                      dataset_source_label='research',
    #                      csv_dir=mesonet_csv_research,
    #                      output_station_tracker_path=tracker_new,
    #                      open_nc_func=open_nc,
    #                      metadata_fields=METADATA,
    #                      input_station_tracker_path=tracker_old,
    #                      time_constraint_yearmonths=None)

    shp = os.path.join(dads, "met", "stations", "madis_17MAY2025_gap.shp")
    check_dir = "/data/ssd2/dads/training/parquet"
    write_stations_to_shapefile(tracker_new, shp, existing_check=check_dir)

    stations = os.path.join(d, "dads", "met", "stations", "dads_stations_elev_mgrs.csv")
    flagged = os.path.join(d, "dads", "met", "stations", "madis_research_flagged.json")
    madis_shapes = os.path.join(d, "climate", "madis", "LDAD", "mesonet", "shapes")
    mgrs_ = os.path.join(d, "boundaries", "mgrs", "MGRS_100km_world.shp")
    # process_weather_stations(stations, madis_shapes, flagged, mgrs_, stations_out)

    shp = os.path.join("/home/dgketchum/Downloads", "madis_shapefile.shp")
    stations_out = os.path.join(
        d, "dads", "met", "stations", "dads_stations_res_elev_mgrs.shp"
    )
    shp_out = os.path.join("/home/dgketchum/Downloads", "madis_28OCT2024.shp")
    # write_missing(shp, stations_out, shp_out)

# ========================= EOF ====================================================================
