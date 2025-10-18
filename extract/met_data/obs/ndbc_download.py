import os
import gzip
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

_STD_INDEX_CACHE = None


def _fetch_stdmet_index():
    global _STD_INDEX_CACHE
    if _STD_INDEX_CACHE is not None:
        return _STD_INDEX_CACHE
    url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    # grab candidate filenames ending with .txt.gz from links or plain text
    files = set(re.findall(r"([A-Za-z0-9_]+\.txt\.gz)", text))
    _STD_INDEX_CACHE = sorted(files)
    return _STD_INDEX_CACHE


def _available_station_years(station_id):
    idx = _fetch_stdmet_index()
    sid = station_id.lower()
    years = []
    for fn in idx:
        fl = fn.lower()
        if fl.startswith(sid + 'h') and fl.endswith('.txt.gz'):
            m = re.search(r"h(\d{4})\.txt\.gz$", fl)
            if m:
                try:
                    years.append(int(m.group(1)))
                except Exception:
                    pass
    years = sorted(list(set(years)))
    return years


def get_ndbc_stations(out_dir, overwrite=False):
    """
    Fetches the list of all NDBC stations, filters for buoys, and saves
    the metadata as a CSV and a shapefile.
    """
    meta_csv = os.path.join(out_dir, 'ndbc_stations.csv')
    meta_shp = os.path.join(out_dir, 'ndbc_stations.shp')

    if not overwrite and os.path.exists(meta_csv) and os.path.exists(meta_shp):
        print('NDBC station metadata files exist, skipping download.')
        return pd.read_csv(meta_csv).set_index('station_id')

    url = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
    print("Downloading NDBC station metadata...")
    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.splitlines()
    header_line = next((ln for ln in lines if ln.startswith('#') and '|' in ln), None)
    if header_line is None:
        raise ValueError('Unexpected NDBC station_table.txt format')
    columns = [c.strip().lstrip('#').strip() for c in header_line.split('|')]
    columns = columns[:9]

    rows = []
    for line in lines:
        if not line or line.startswith('#') or '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        parts = parts[:9]
        if len(parts) < len(columns):
            parts += [''] * (len(columns) - len(parts))
        rows.append(parts[:len(columns)])

    df = pd.DataFrame(rows, columns=columns)
    if 'STATION_ID' in df.columns:
        df.rename(columns={'STATION_ID': 'station_id'}, inplace=True)
    if 'LOCATION' in df.columns:
        df.rename(columns={'LOCATION': 'location'}, inplace=True)

    # Parse LOCATION column (decimal or DMS inside parentheses)
    coords = df['location'].str.extract(r'([+-]?\d+(?:\.\d+)?)\s*(N|S)\s*([+-]?\d+(?:\.\d+)?)\s*(W|E)')
    lat = pd.to_numeric(coords[0], errors='coerce')
    lon = pd.to_numeric(coords[2], errors='coerce')
    lat = lat * np.where(coords[1] == 'S', -1, 1)
    lon = lon * np.where(coords[3] == 'W', -1, 1)

    def _dms_fallback(s):
        if not isinstance(s, str):
            return np.nan, np.nan
        m = re.search(r"(\d+)[^\d]+(\d+)[^\d]+(\d+).*?([NS]).*?(\d+)[^\d]+(\d+)[^\d]+(\d+).*?([WE])", s)
        if not m:
            return np.nan, np.nan
        d1, m1, s1, hemi1, d2, m2, s2, hemi2 = m.groups()
        try:
            d1 = float(d1);
            m1 = float(m1);
            s1 = float(s1)
            d2 = float(d2);
            m2 = float(m2);
            s2 = float(s2)
        except Exception:
            return np.nan, np.nan
        lat_dd = d1 + m1 / 60.0 + s1 / 3600.0
        lon_dd = d2 + m2 / 60.0 + s2 / 3600.0
        if hemi1.upper() == 'S':
            lat_dd = -lat_dd
        if hemi2.upper() == 'W':
            lon_dd = -lon_dd
        return lat_dd, lon_dd

    need_fallback = lat.isna() | lon.isna()
    if need_fallback.any():
        fallback_vals = df.loc[need_fallback, 'location'].apply(_dms_fallback)
        if len(fallback_vals) > 0:
            fb_lat = [t[0] for t in fallback_vals]
            fb_lon = [t[1] for t in fallback_vals]
            lat.loc[need_fallback] = fb_lat
            lon.loc[need_fallback] = fb_lon

    df['latitude'] = lat
    df['longitude'] = lon
    df = df[~df['latitude'].isna() & ~df['longitude'].isna()]

    print(f"Found {len(df)} NDBC buoys.")

    df.to_csv(meta_csv, index=False)
    print(f"Saved station metadata to {meta_csv}")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    gdf.to_file(meta_shp)
    print(f"Saved station shapefile to {meta_shp}")

    return df.set_index('station_id')


def download_ndbc_station_data(station_id, data_dst, start_year=1970, overwrite=False):
    """
    Downloads and processes all historical standard meteorological data for a single NDBC station.
    """
    out_file = os.path.join(data_dst, f'{station_id}.parquet')
    if not overwrite and os.path.exists(out_file):
        return station_id, 0

    base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    latest_base = "https://www.ndbc.noaa.gov/data/l_stdmet/"
    current_year = datetime.now().year
    station_dfs = []
    from_index = _available_station_years(station_id)
    years = sorted(from_index, reverse=True) if from_index else list(range(start_year, current_year + 1))[::-1]

    years_404_streak = 0
    for year in years:
        fname = f"{station_id.lower()}h{year}.txt.gz"
        url = base_url + fname
        try:
            resp = requests.get(url, timeout=30)
            # 'https://www.ndbc.noaa.gov/data/historical/stdmet/npsf1h2018.txt.gz'
            if resp.status_code == 404:
                years_404_streak += 1
                if years_404_streak >= 5:
                    break
                continue
            resp.raise_for_status()
            years_404_streak = 0
            with gzip.open(io.BytesIO(resp.content), 'rt') as f:
                raw = f.read()
            lines = raw.splitlines()
            hdr = next((ln for ln in lines if ln.startswith('#') and 'YY' in ln and 'MM' in ln), None)
            if hdr:
                names_raw = hdr.lstrip('#').split()
                tmp = pd.read_csv(io.StringIO(raw), sep=r'\s+', comment='#', header=None,
                                  names=names_raw,
                                  na_values=['99', '99.0', '999', '999.0', '9999.0'], engine='python')
            else:
                tmp = pd.read_csv(io.StringIO(raw), sep=r'\s+', comment='#', header=None,
                                  na_values=['99', '99.0', '999', '999.0', '9999.0'], engine='python')
            if tmp.empty:
                continue
            ncols = tmp.shape[1]
            std = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD',
                   'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

            if hdr is None:
                names = std[:min(ncols, len(std))]
                tmp = tmp.iloc[:, :len(names)]
                tmp.columns = names

            if 'mm' not in tmp.columns:
                tmp['mm'] = 0

            d = tmp[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'})
            tmp.index = pd.to_datetime(d, errors='coerce')
            tmp = tmp[~tmp.index.isna()]
            station_dfs.append(tmp)
        except requests.exceptions.RequestException:
            continue
        except Exception:
            continue

    # append most recent (latest) file
    fname = f"{station_id.lower()}.txt"
    url = latest_base + fname
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
        lines = text.splitlines()
        hdr = next((ln for ln in lines if ln.startswith('#') and 'YY' in ln and 'MM' in ln), None)
        if hdr:
            names_raw = hdr.lstrip('#').split()
            tmp = pd.read_csv(io.StringIO(text), sep=r'\s+', comment='#', header=None,
                              names=names_raw,
                              na_values=['99', '99.0', '999', '999.0', '9999.0'], engine='python')
        else:
            tmp = pd.read_csv(io.StringIO(text), sep=r'\s+', comment='#', header=None,
                              na_values=['99', '99.0', '999', '999.0', '9999.0'], engine='python')
        if 'mm' not in tmp.columns:
            tmp['mm'] = 0

        d = tmp[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
            columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'})
        tmp.index = pd.to_datetime(d, errors='coerce')
        tmp = tmp[~tmp.index.isna()]
        if not tmp.empty:
            station_dfs.append(tmp)

    except requests.exceptions.RequestException:
        pass
    except Exception:
        pass

    if not station_dfs:
        return station_id, 0

    station_df = pd.concat(station_dfs)
    station_df = station_df.sort_index()
    station_df = station_df[~station_df.index.duplicated(keep='first')]
    # Drop original date part columns to avoid Arrow dtype issues
    for c in ['YY', 'YYYY', 'MM', 'DD', 'hh', 'mm']:
        if c in station_df.columns:
            station_df.drop(columns=[c], inplace=True)

    rename_map = {
        'WDIR': 'wind_dir', 'WSPD': 'wind_speed', 'GST': 'wind_gust', 'WVHT': 'wave_height',
        'DPD': 'dominant_wave_period', 'APD': 'average_wave_period', 'MWD': 'mean_wave_dir',
        'PRES': 'pressure', 'ATMP': 'air_temp', 'WTMP': 'water_temp', 'DEWP': 'dewpoint',
        'VIS': 'visibility', 'TIDE': 'tide'
    }
    station_df.rename(columns=rename_map, inplace=True)

    # Coerce expected numeric columns to numeric (handles strings like 'VRB', 'MM')
    numeric_cols = ['wind_dir', 'wind_speed', 'wind_gust', 'wave_height', 'dominant_wave_period',
                    'average_wave_period', 'mean_wave_dir', 'pressure', 'air_temp', 'water_temp',
                    'dewpoint', 'visibility', 'tide']
    for c in numeric_cols:
        if c in station_df.columns:
            station_df[c] = pd.to_numeric(station_df[c], errors='coerce')

    station_df.to_parquet(out_file)
    return station_id, len(station_df)


def process_all_stations(station_ids, data_dst, start_year, workers, overwrite, debug=False):
    """
    Processes all NDBC stations in parallel using a process pool.
    """
    print(f"Processing {len(station_ids)} stations with {workers} workers...")
    tasks = {station: (station, data_dst, start_year, overwrite) for station in station_ids}

    if debug or workers <= 1:
        for station in tqdm(station_ids, desc="Downloading NDBC Records (seq)"):
            sid, nrec = download_ndbc_station_data(*tasks[station])
            if nrec > 0:
                tqdm.write(f"Successfully processed station {sid} with {nrec} records.")

        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_ndbc_station_data, *args): station for station, args in tasks.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading NDBC Records"):
            try:
                station_id, record_count = future.result()
                if record_count > 0:
                    tqdm.write(f"Successfully processed station {station_id} with {record_count} records.")
            except Exception as e:
                station_id = futures[future]
                tqdm.write(f"Station {station_id} failed with an exception: {e}")


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    ndbc = os.path.join(d, 'climate', 'ndbc')

    ndbc_meta_dir = os.path.join(ndbc, 'ndbc_meta')
    ndbc_data_dir = os.path.join(ndbc, 'ndbc_records')

    workers = 4
    debug_run = False
    overwrite_station_list = False
    overwrite_station_data = False
    start_year_download = 1970

    meta_pth = os.path.join(ndbc_meta_dir, 'ndbc_stations.csv')
    if not overwrite_station_list and os.path.exists(meta_pth):
        gdf = pd.read_csv(meta_pth)
        stations_df = gdf.set_index('station_id')
    else:
        stations_df = get_ndbc_stations(ndbc_meta_dir, overwrite=overwrite_station_list)

    station_ids = stations_df.index.tolist()
    # station_ids = ['dpia1']

    process_all_stations(station_ids,
                         data_dst=ndbc_data_dir,
                         start_year=start_year_download,
                         workers=1,
                         overwrite=overwrite_station_data,
                         debug=debug_run)

# ========================= EOF ====================================================================
