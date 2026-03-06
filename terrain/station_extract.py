import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

COLS = [
    "latitude",
    "longitude",
    "aspect",
    "elevation",
    "slope",
    "tpi_10000",
    "tpi_22500",
    "tpi_2500",
    "tpi_500",
]


def _process_csv_file(args):
    path, station_out, index_col, coords, overwrite = args

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        os.remove(path)
        print(f"found empty csv at {path}")
        return

    df = df.rename(columns={k: v for k, v in zip(coords, ["latitude", "longitude"])})
    for _, r in df.iterrows():
        sf = os.path.join(station_out, f"{r[index_col]}.csv")
        if os.path.exists(sf) and not overwrite:
            continue
        d = r[COLS + [index_col]]
        d.to_csv(sf)


def write_station_terrain(
    tile_dirs,
    station_out,
    shuffle=False,
    overwrite=False,
    index_col="fid",
    coords=["latitude", "longitude"],
    num_workers=12,
):

    if isinstance(tile_dirs, str):
        tile_dirs = [tile_dirs]

    csv_files = []
    for tdir in tile_dirs:
        if os.path.isdir(tdir):
            csv_files.extend(
                [os.path.join(tdir, f) for f in os.listdir(tdir) if f.endswith(".csv")]
            )

    if shuffle and csv_files:
        csv_files = pd.Series(csv_files).sample(frac=1, random_state=42).tolist()

    jobs = [(p, station_out, index_col, coords, overwrite) for p in csv_files]

    if num_workers is None or int(num_workers) <= 1:
        for job in tqdm(jobs, total=len(jobs), desc="Writing terrain"):
            _process_csv_file(job)
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
            for _ in tqdm(
                ex.map(_process_csv_file, jobs), total=len(jobs), desc="Writing terrain"
            ):
                pass


if __name__ == "__main__":
    d = "/media/research/IrrigationGIS/dads"
    if not os.path.exists(d):
        d = "/nas/dads"

    out = os.path.join(d, "dem", "terrain", "station_data")

    dirs_ = [
        # os.path.join(d, 'dem', 'terrain', 'madis_stations'),
        # os.path.join(d, 'dem', 'terrain', 'new_madis'),
        # os.path.join(d, 'dem', 'terrain', 'dads_10FEB'),
        os.path.join(d, "dem", "terrain", "madis_17MAY2025_gap_mgrs"),
        # os.path.join(d, 'dem', 'terrain', 'ndbc_stations'),  # NDBC tile CSVs
    ]
    write_station_terrain(
        dirs_,
        out,
        shuffle=False,
        overwrite=False,
        index_col="fid",
        coords=["latitude", "longitude"],
        num_workers=12,
    )

    # ghcn_dir_ = os.path.join(d, 'dem', 'terrain', 'ghcn_stations')
    # write_station_terrain(ghcn_dir_, out, shuffle=False, overwrite=False,
    #                       index_col='STAID', coords=['LAT', 'LON'], num_workers=12)

# ========================= EOF ====================================================================
