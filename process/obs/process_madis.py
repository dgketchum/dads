import concurrent.futures
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from timezonefinder import TimezoneFinder
import pyarrow

from utils.calc_eto import calc_asce_params
from utils.station_parameters import station_par_map


def process_daily_data(hourly_df, rsun_data, lat_, elev_, zw_=2.0, qaqc=False):
    hourly_df['date'] = hourly_df.index.date
    hourly_df['hour'] = hourly_df.index.hour

    valid_obs_count = hourly_df[['date']].groupby('date').agg({'date': 'count'}).copy()
    if np.nanmax(valid_obs_count['date']) > 24:
        agg_rules = {
            'temperature': 'mean',
            'relHumidity': 'mean',
            'rsds': 'mean',
            'ea': 'mean',
            'wind': 'mean',
            'wind_dir': 'mean',
            'doy': 'first',
            'precipAccum': 'first',  # Keep these for the custom precip function
            'precipAccumPeriod': 'first'
        }
        valid_agg_rules = {k: v for k, v in agg_rules.items() if k in hourly_df.columns}

        hourly_df = hourly_df.groupby(['date', 'hour']).agg(**valid_agg_rules)
        hourly_df.reset_index(inplace=True)
        if 'date' in hourly_df.columns and 'hour' in hourly_df.columns:
            hourly_df['datetime'] = pd.to_datetime(
                hourly_df['date'].astype(str) + ' ' + hourly_df['hour'].astype(str) + ':00:00', errors='coerce')
            hourly_df.set_index('datetime', inplace=True)
            hourly_df.drop(columns=['date', 'hour'], inplace=True)

    # U is the velocity toward east and V is the velocity toward north
    wind_direction_rad = np.deg2rad(hourly_df['wind_dir'])
    hourly_df['u'] = hourly_df['wind'] * (-np.sin(wind_direction_rad))
    hourly_df['v'] = hourly_df['wind'] * (-np.cos(wind_direction_rad))
    if hourly_df.empty:
        return None

    hourly_df['date'] = hourly_df.index.date
    valid_obs_count = hourly_df[['date']].groupby('date').agg({'date': 'count'}).copy()

    daily_agg_rules = {
        'tmax': ('temperature', 'max'),
        'tmin': ('temperature', 'min'),
        'tmean': ('temperature', 'mean'),
        'rsds': ('rsds', 'sum'),
        'ea': ('ea', 'mean'),
        'wind': ('wind', 'mean'),
        'wind_dir': ('wind_dir', 'mean'),
        'u': ('u', 'mean'),
        'v': ('v', 'mean'),
        'doy': ('doy', 'first')
    }
    valid_daily_agg_rules = {k: v for k, v in daily_agg_rules.items() if v in hourly_df.columns}
    daily_df = hourly_df.groupby('date').agg(**valid_daily_agg_rules).copy()

    if 'precipAccum' in hourly_df.columns and 'precipAccumPeriod' in hourly_df.columns:
        daily_prcp = hourly_df.groupby('date').apply(calculate_daily_precipitation).to_frame('prcp')
        daily_df = daily_df.join(daily_prcp)
    else:
        daily_df['prcp'] = np.nan

    daily_df['obs_ct'] = valid_obs_count
    daily_df = daily_df[daily_df['obs_ct'] >= 18]

    if daily_df.empty:
        return None

    daily_df.drop(columns=['obs_ct'], inplace=True)
    daily_df.index = pd.DatetimeIndex(daily_df.index)

    if daily_df.empty:
        return None

    if qaqc:
        if not all(col in daily_df.columns for col in ['doy', 'ea', 'rsds']):
            return daily_df

        daily_df['month'] = daily_df.index.month
        daily_df['rsun'] = daily_df['doy'].map(rsun_data)
        daily_df.loc[daily_df['rsun'].isna(), 'rsun'] = 0

        if daily_df['rsds'].notna().any():
            daily_df['rolling_rsds_max'] = daily_df['rsds'].rolling(15, min_periods=1).max()
            daily_df['rolling_rsds_max'] = daily_df['rolling_rsds_max'].bfill().ffill()
            daily_df.loc[
                daily_df['rolling_rsds_max'].isna() | (daily_df['rolling_rsds_max'] == 0), 'rolling_rsds_max'] = np.nan
            daily_df['rsds'] *= daily_df['rsun'] / daily_df['rolling_rsds_max']
        else:
            daily_df['rsds'] = np.nan

    required_asce_cols = ['tmax', 'tmin', 'rsds', 'wind', 'ea']
    daily_df.rename(columns={'tmax': 'max_temp', 'tmin': 'min_temp'}, inplace=True)

    if not all(col in daily_df.columns for col in required_asce_cols):
        return daily_df

    asce_params = daily_df.apply(calc_asce_params, lat=lat_, elev=elev_, zw=zw_, axis=1)

    if not asce_params.empty:
        asce_df = pd.DataFrame(asce_params.tolist(), index=daily_df.index)
        expected_cols = ['mean_temp', 'vpd', 'rn', 'u2', 'eto']
        asce_df.columns = expected_cols[:len(asce_df.columns)]
        for col in asce_df.columns:
            if col in expected_cols:
                daily_df[col] = asce_df[col]

    return daily_df


def apply_madis_qc(df):
    acceptable_qc_flags = ['S', 'V']
    qc_vars = {'temperature': float,
               'dewpoint': float,
               'relHumidity': float,
               'windSpeed': float,
               'windDir': float,
               'precipAccum': float,
               'solarRadiation': float}

    for var in qc_vars:
        dd_col = f'{var}DD'
        if var in df.columns and dd_col in df.columns:
            df.loc[~df[dd_col].isin(acceptable_qc_flags), var] = np.nan

    return df


def calculate_daily_precipitation(daily_group):
    # Prioritize 24-hour accumulation reports if they exist.
    # Codes -1, -2, -3 represent various 24-hour totals.
    daily_reports = daily_group[daily_group['precipAccumPeriod'].isin([-1, -2, -3])]
    if not daily_reports.empty:
        # Return the maximum reported 24-hour value for the day.
        return daily_reports['precipAccum'].max()

    # If no 24-hour reports, sum the hourly reports.
    # Code 1 indicates the value is for the "last 1 hr".
    hourly_reports = daily_group[daily_group['precipAccumPeriod'] == 1]
    if not hourly_reports.empty:
        return hourly_reports['precipAccum'].sum()

    # Handle other accumulation periods (3, 6, 12 hr) if necessary,
    # or return NaN if no usable data is found.
    # For simplicity, this example prioritizes 24h and 1h reports.
    return np.nan


def process_single_station(fid, madis_src, madis_dst, rsun_tables, overwrite, qaqc, plot, alt_src):
    """
    Processes all data for a single station by first concatenating all monthly
    files and then processing the combined dataframe.
    """
    out_file = os.path.join(madis_dst, '{}.parquet'.format(fid))

    if os.path.exists(out_file) and not overwrite:
        return

    try:
        station_dir = os.path.join(madis_src, fid)
    except TypeError:
        print(fid, 'no_dir')
        return

    if not os.path.isdir(station_dir):
        if alt_src:
            station_dir = os.path.join(alt_src, fid)
        if not os.path.isdir(station_dir):
            print(fid, 'Source directory not found')
            return

    valid_files = [f for f in glob.glob(os.path.join(station_dir, '*.parquet.gz'))]

    if not valid_files:
        print(fid, 'no_files')
        return

    rsun_file = os.path.join(rsun_tables, '{}.csv'.format(fid))
    if not os.path.exists(rsun_file):
        print(fid, 'error', f'RSun file missing: {rsun_file}')
        return

    rsun = pd.read_csv(rsun_file, index_col=0)
    if fid not in rsun.columns:
        print(fid, 'error', f'RSun file missing {fid}: {rsun_file}', flush=True)
        return

    rsun = (rsun[fid] * 0.0036)

    if np.any(np.isnan(rsun.values)):
        print(f"Station {fid} has nan in RSun data, removing {rsun_file}")
        os.remove(rsun_file)
        return

    rsun = rsun.to_dict()

    df_list = []
    for file_ in valid_files:
        try:
            df_list.append(pd.read_parquet(file_))
        except (pd.errors.EmptyDataError, pyarrow.lib.ArrowInvalid):
            print(f'file {file_} is invalid or empty, skipping')
            continue

    if not df_list:
        print(f'{fid}: No valid data files found.')
        return

    c = pd.concat(df_list)
    c.index = pd.to_datetime(c['datetime'])
    c = c.drop(columns=['datetime'])

    if c.empty:
        print(f'{fid}: Combined dataframe is empty.')
        return

    if qaqc:
        c = apply_madis_qc(c)

    c = c.sort_index()

    try:
        lon = c['longitude'].iloc[0]
        lat = c['latitude'].iloc[0]
        elv = c['elevation'].iloc[0]
    except (KeyError, IndexError) as e:
        print(f'{fid}: Failed to extract metadata. Ensure "longitude", "latitude", and "elevation" exist. Error: {e}')
        return

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lon, lat=lat)
    if timezone_str is None:
        print(f'{fid}: Could not determine timezone for coords', flush=True)
        return
    local_tz = pytz.timezone(timezone_str)

    c['doy'] = c.index.dayofyear
    if c.index.tz is None:
        c.index = c.index.tz_localize('UTC')
    c.index = c.index.tz_convert(local_tz)

    required_cols = ['temperature', 'relHumidity', 'solarRadiation', 'windSpeed', 'windDir', 'precipAccum']
    for col in required_cols:
        if col not in c.columns:
            c[col] = np.nan

    c['temperature'] -= 273.15
    es = 0.6108 * np.exp(17.27 * c['temperature'] / (c['temperature'] + 237.3))
    c['ea'] = (c['relHumidity'] / 100) * es
    c['rsds'] = c['solarRadiation'] * 0.0036
    c['wind'] = c['windSpeed']
    c['wind_dir'] = c['windDir']

    df = process_daily_data(c, rsun, lat_=lat, elev_=elv, zw_=2.0, qaqc=qaqc)

    if df is None or df.empty:
        print(f'{fid}: No processable records found after daily aggregation', flush=True)
        return

    if plot:
        raise NotImplementedError

    df = df.sort_index()
    if df.index.has_duplicates:
        # It's possible for daily data to have duplicates if source files overlap.
        # We can keep the first entry.
        df = df[~df.index.duplicated(keep='first')]
        print(f'{fid} had duplicates, kept first entry.')

    # Add metadata to the final daily dataframe before saving
    df['latitude'] = lat
    df['longitude'] = lon
    df['elevation'] = elv

    df.to_parquet(out_file)
    obs_ct = df.shape[0]
    print(f'{fid}: successfully processed {obs_ct} daily observations')
    return


def read_hourly_data(madis_src, madis_dst, rsun_tables, overwrite=False,
                     qaqc=False, plot=None, alt_src=None, n_workers=4, debug=False):
    """
    Scans a directory for station data, and processes each station.
    This function no longer requires a station metadata file.
    """
    try:
        station_fids = [d for d in os.listdir(madis_src)]
    except FileNotFoundError:
        print(f"Error: Source directory not found at {madis_src}")
        return

    if not station_fids:
        print(f"No station subdirectories found in {madis_src}")
        return

    total_stations_to_process = len(station_fids)
    print(f"Found {total_stations_to_process} stations to process.")

    if debug:
        for fid in station_fids:
            if fid != 'COVM':
                continue
            process_single_station(fid, madis_src, madis_dst, rsun_tables, overwrite, qaqc, plot, alt_src)
    else:
        if n_workers is None:
            n_workers = os.cpu_count()
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for fid in station_fids:
                future = executor.submit(process_single_station, fid, madis_src, madis_dst, rsun_tables,
                                         overwrite, qaqc, plot, alt_src)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                pass


def plot_daily_data(pdf, station_id, year, out_fig):
    if not isinstance(pdf, pd.DataFrame):
        pdf = pd.read_csv(pdf, index_col=0, parse_dates=True)

    start_date = '{}-01-01'.format(year)
    end_date = '{}-12-31'.format(year)

    all_dates = pd.DatetimeIndex(pd.date_range(start_date, end_date))
    pdf = pdf.reindex(all_dates)
    pdf['doy'] = pdf.index.dayofyear
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 16), sharex=True)

    sns.barplot(data=pdf, x='doy', y='prcp', ax=axes[0])
    axes[0].set_ylabel("Precipitation (mm)")
    axes[0].set_xlim(1, 366)

    for i, var in enumerate(['vpd', 'rsds', 'u2', 'mean_temp', 'eto'], start=1):
        sns.lineplot(data=pdf, x='doy', y=var, ax=axes[i])
        if var == 'rsds':
            sns.lineplot(data=pdf, x='doy', y='rso', ax=axes[i])
            sns.lineplot(data=pdf, x='doy', y='rsun', ax=axes[i])
        axes[i].set_ylabel(var)
        axes[i].set_xlim(1, 366)

    axes[i].set_xticks([0, 100, 200, 300, 365])

    axes[-1].set_xlabel("Day of Year")
    fig.suptitle(f"Daily Meteorological Data for Station {station_id} - {year}", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(out_fig)
    plt.close()
    print(os.path.basename(out_fig))


def write_daily_maids_plots(madis_daily_dir, plot_dir, target_sites=None):
    files_ = list(os.listdir(madis_daily_dir))

    for f in files_:

        file_ = os.path.join(madis_daily_dir, f)
        site = f.split('.')[0]

        if target_sites:
            if site not in target_sites:
                continue

        search_pattern = os.path.join(plot_dir.format('checked'), f'{site}_*.png')
        matching_files = glob.glob(search_pattern)
        if len(matching_files) > 0:
            print('{} has been checked'.format(site))
            continue

        search_pattern = os.path.join(plot_dir.format('to_check'), f'{site}_*.png')
        written_files = glob.glob(search_pattern)
        if len(written_files) > 0:
            print('{} has been written'.format(site))
            continue

        df = pd.read_csv(file_, index_col=0, parse_dates=True)
        years = np.unique(df.index.year)

        for year in years:
            c = df[df.index.year == year]
            out_ = os.path.join(plot_dir.format('to_check'), f'{site}_{year}.png')
            plot_daily_data(c, site, year=year, out_fig=out_)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    madis_hourly_public = os.path.join(d, 'climate', 'madis', 'LDAD_public', 'mesonet', 'csv')
    madis_hourly_research = '/data/ssd2/madis/extracts_qaqc'
    madis_pst = '/data/ssd2/madis/pst'

    import json

    pst_files = [os.path.join(madis_pst, f) for f in os.listdir(madis_pst) if f.endswith('.json')]
    tables = ['code1PST', 'code2PST', 'code3PST', 'code4PST', 'typePST', 'namePST']
    pst_dct = {}
    first = True

    for f in sorted(pst_files):
        with open(f, 'r') as fp:
            dct = json.load(fp)

        for t in tables:
            if 'code' in t:
                dct[t] = [int(float(i)) for i in dct[t] if i != 'nan']
            else:
                dct[t] = [i.strip() for i in dct[t] if i != '']
        print(os.path.basename(f), len(dct['namePST']))

        for idx, k in enumerate(dct['namePST']):
            if k == 'nan':
                continue
            if k not in pst_dct:
                pst_dct[k] = {t: dct[t][idx] for t in tables[:-1]}
            else:
                for kk, vv in pst_dct[k].items():
                    try:
                        assert pst_dct[k][kk] == dct[kk][idx]
                    except AssertionError:
                        print(f'{kk} mismatch: {pst_dct[k][kk]}, {dct[kk][idx]}')

    madis_daily_ = '/data/ssd2/madis/daily'

    solrad = os.path.join(d, 'dads', 'dem', 'rsun_stations')

    read_hourly_data(madis_hourly_research, madis_daily_, solrad,
                     overwrite=True, qaqc=True, plot=None, debug=True,
                     alt_src=madis_hourly_public, n_workers=25)

# ========================= EOF ====================================================================
