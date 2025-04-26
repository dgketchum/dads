import concurrent.futures
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from timezonefinder import TimezoneFinder

from utils.calc_eto import calc_asce_params
from utils.station_parameters import station_par_map


def process_daily_data(hourly_df, rsun_data, lat_, elev_, zw_=2.0, qaqc=False):
    hourly_df['date'] = hourly_df.index.date
    hourly_df['hour'] = hourly_df.index.hour

    valid_obs_count = hourly_df[['date']].groupby('date').agg({'date': 'count'}).copy()
    if np.nanmax(valid_obs_count['date']) > 24:
        hourly_df = hourly_df.groupby(['date', 'hour']).agg(
            temperature=('temperature', 'mean'),
            precipAccum=('precipAccum', 'mean'),
            relHumidity=('relHumidity', 'mean'),
            rsds=('rsds', 'mean'),
            ea=('ea', 'mean'),
            wind=('wind', 'mean'),
            wind_dir=('wind_dir', 'mean'),
            doy=('doy', 'first'),
        )
        hourly_df.reset_index(inplace=True)
        if 'date' in hourly_df.columns and 'hour' in hourly_df.columns:
            hourly_df['datetime'] = pd.to_datetime(
                hourly_df['date'].astype(str) + ' ' + hourly_df['hour'].astype(str) + ':00:00', errors='coerce')
            hourly_df.set_index('datetime', inplace=True)
            hourly_df.drop(columns=['date', 'hour'], inplace=True)

    wind_direction_rad = np.deg2rad(hourly_df['wind_dir'])
    hourly_df['u'] = hourly_df['wind'] * (-np.sin(wind_direction_rad))
    hourly_df['v'] = hourly_df['wind'] * (-np.cos(wind_direction_rad))

    hourly_df['precip_increment'] = hourly_df['precipAccum'].diff()
    hourly_df.loc[hourly_df['precip_increment'] < 0, 'precip_increment'] = 0
    hourly_df.loc[hourly_df['precip_increment'] > 50., 'precip_increment'] = 0

    if qaqc:
        hourly_df.loc[hourly_df['rsds'] < 0, 'rsds'] = np.nan
        hourly_df.loc[hourly_df['rsds'] > 1500., 'rsds'] = np.nan
        hourly_df.loc[hourly_df['wind'] < 0., 'wind'] = np.nan
        hourly_df.loc[hourly_df['wind'] > 200., 'wind'] = np.nan
        hourly_df.loc[hourly_df['temperature'] > 57.22, 'temperature'] = np.nan
        hourly_df.loc[hourly_df['temperature'] < -59.44, 'temperature'] = np.nan
        hourly_df.loc[hourly_df['relHumidity'] < 0.0, 'relHumidity'] = np.nan
        hourly_df.loc[hourly_df['relHumidity'] < 0.0, 'ea'] = np.nan
        hourly_df.loc[hourly_df['relHumidity'] > 100.0, 'relHumidity'] = np.nan
        hourly_df.loc[hourly_df['relHumidity'] > 100.0, 'ea'] = np.nan

    if hourly_df.empty:
        return None

    hourly_df['date'] = hourly_df.index.date
    valid_obs_count = hourly_df[['date']].groupby('date').agg({'date': 'count'}).copy()

    daily_df = hourly_df.groupby('date').agg(
        tmax=('temperature', 'max'),
        tmin=('temperature', 'min'),
        tmean=('temperature', 'mean'),
        prcp=('precip_increment', 'sum'),
        rsds=('rsds', 'sum'),
        ea=('ea', 'mean'),
        wind=('wind', 'mean'),
        wind_dir=('wind_dir', 'mean'),
        u=('u', 'mean'),
        v=('v', 'mean'),
        doy=('doy', 'first')
    ).copy()

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

    required_asce_cols = ['max_temp', 'min_temp', 'rsds', 'wind', 'ea']
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


def process_single_station(fid, row, kw, madis_src, madis_dst, rsun_tables, overwrite, qaqc, plot, alt_src):
    """"""
    lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
    out_file = os.path.join(madis_dst, '{}.parquet'.format(fid))

    if os.path.exists(out_file) and not overwrite:
        return fid, 'exists', 'File exists', 0

    station_dir = os.path.join(madis_src, fid)
    if not os.path.isdir(station_dir):
        if alt_src:
            station_dir = os.path.join(alt_src, fid)
        if not os.path.isdir(station_dir):
            return fid, 'no_dir', 'Source directory not found', 0

    files_ = [os.path.join(station_dir, f) for f in os.listdir(station_dir)]
    years = []
    valid_files = []
    for f in files_:
        year = int(os.path.basename(f).split('.')[0].split('_')[-1])
        years.append(year)
        valid_files.append(f)

    if not valid_files:
        return fid, 'no_files', 'No valid data files found', 0

    rsun_file = os.path.join(rsun_tables, '{}.csv'.format(fid))
    if not os.path.exists(rsun_file):
        return fid, 'error', f'RSun file missing: {os.path.basename(rsun_file)}', 0

    rsun = pd.read_csv(rsun_file, index_col=0)
    if fid not in rsun.columns:
        raise KeyError(f"Station {fid} not found in RSun table {os.path.basename(rsun_file)}")
    rsun = (rsun[fid] * 0.0036)
    if np.any(np.isnan(rsun.values)):
        raise ValueError("NaN values found in RSun data")
    rsun = rsun.to_dict()

    df, first, local_tz = None, True, None
    tf = TimezoneFinder()

    for file_, yr in zip(valid_files, years):
        try:
            c = pd.read_csv(file_, index_col=0, parse_dates=True, low_memory=False)
        except pd.errors.EmptyDataError:
            continue

        if c.empty:
            continue

        if not isinstance(c.index, pd.DatetimeIndex):
            continue

        c = c.sort_index()
        c['doy'] = c.index.dayofyear

        if first:
            timezone_str = tf.timezone_at(lng=lon, lat=lat)
            if timezone_str is None:
                return fid, 'error', f"Could not determine timezone for coords ({lat}, {lon})", 0
            local_tz = pytz.timezone(timezone_str)

        if c.index.tz is None:
            c.index = c.index.tz_localize('GMT')
        c.index = c.index.tz_convert(local_tz)

        required_cols = ['temperature', 'relHumidity', 'solarRadiation', 'windSpeed', 'windDir', 'precipAccum']
        if not any(col in c.columns for col in required_cols):
            continue

        for col in required_cols:
            if col not in c.columns:
                c[col] = np.nan

        c['temperature'] -= 273.15
        es = 0.6108 * np.exp(17.27 * c['temperature'] / (c['temperature'] + 237.3))
        c['ea'] = (c['relHumidity'] / 100) * es
        c['rsds'] = c['solarRadiation'] * 0.0036
        c['wind'] = c['windSpeed']
        c['wind_dir'] = c['windDir']

        c_processed = process_daily_data(c, rsun, lat_=lat, elev_=elv, zw_=2.0, qaqc=qaqc)

        if c_processed is None or c_processed.empty:
            continue

        if plot and not c_processed.empty:
            out_plot_dir = plot.format('to_check')
            os.makedirs(out_plot_dir, exist_ok=True)
            out_plot_file = os.path.join(out_plot_dir, '{}_{}.png'.format(fid, yr))
            # plot_daily_data(c_processed, fid, yr, out_plot_file) # Assumed external function

        if first:
            df = c_processed.copy()
            first = False
        else:
            df_cols = set(df.columns)
            c_cols = set(c_processed.columns)
            if df_cols != c_cols:
                all_cols = df_cols.union(c_cols)
                df = df.reindex(columns=all_cols)
                c_processed = c_processed.reindex(columns=all_cols)
            df = pd.concat([df, c_processed], ignore_index=False, axis=0)

    if df is None or df.empty:
        return fid, 'nodata', 'No processable records found across all years', 0

    df = df.sort_index()
    df.to_parquet(out_file)
    obs_ct = df.shape[0]
    return fid, 'success', f'Wrote {obs_ct} records', obs_ct


def read_hourly_data(stations, madis_src, madis_dst, rsun_tables, shuffle=False, bounds=None, overwrite=False,
                     qaqc=False, plot=None, alt_src=None, stype='madis', n_workers=4, debug=False):
    kw = station_par_map(stype)
    station_list = pd.read_csv(stations, index_col=kw['index'])

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list[kw['lat']] < n) & (station_list[kw['lat']] >= s)]
        station_list = station_list[(station_list[kw['lon']] < e) & (station_list[kw['lon']] >= w)]
    else:
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list[kw['lat']] < n) & (station_list[kw['lat']] >= s)]
        station_list = station_list[(station_list[kw['lon']] < e) & (station_list[kw['lon']] >= w)]

    total_stations_to_process = station_list.shape[0]
    if total_stations_to_process == 0:
        return

    processed_count = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    nodata_count = 0
    total_obs_ct = 0
    exists_count = 0

    os.makedirs(madis_dst, exist_ok=True)
    no_files = []

    if debug:
        for i, (fid, row) in enumerate(station_list.iterrows(), start=1):

            # if fid != 'COVM':
            #     continue

            result = process_single_station(fid, row, kw, madis_src, madis_dst, rsun_tables, overwrite, qaqc, plot,
                                            alt_src)
            processed_count += 1
            _fid, status, message, obs_count = result
            total_obs_ct += obs_count
            if status == 'success':
                success_count += 1
            elif status == 'no_dir':
                skipped_count += 1
            elif status == 'no_files':
                skipped_count += 1
                no_files.append(fid)

            elif status == 'exists':
                exists_count += 1
            elif status == 'error':
                error_count += 1
            elif status == 'nodata':
                nodata_count += 1

            print(f'{fid}: {status}')
            if len(no_files) >= 100:
                print(no_files)
                break

    else:
        if n_workers is None:
            n_workers = os.cpu_count()
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            for fid, row in station_list.iterrows():
                future = executor.submit(process_single_station, fid, row, kw, madis_src, madis_dst, rsun_tables,
                                         overwrite, qaqc, plot, alt_src)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                processed_count += 1
                result = future.result()
                _fid, status, message, obs_count = result
                total_obs_ct += obs_count

                if status == 'success':
                    success_count += 1
                elif status == 'no_dir':
                    skipped_count += 1
                elif status == 'no_files':
                    skipped_count += 1
                elif status == 'exists':
                    exists_count += 1
                elif status == 'error':
                    error_count += 1
                elif status == 'nodata':
                    nodata_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total stations attempted: {total_stations_to_process}")
    print(f"Successfully processed:   {success_count}")
    print(f"Skipped (no src dir/no files):  {skipped_count}")
    print(f"No processable data:    {nodata_count}")
    print(f"File exists:    {exists_count}")
    print(f"Errors encountered:     {error_count}")
    print(f"Total observations written: {total_obs_ct}")
    print("-------------------------\n")


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

    # pandarallel.initialize(nb_workers=6)

    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations_10FEB2025.csv')

    madis_hourly_public = os.path.join(d, 'climate', 'madis', 'LDAD_public', 'mesonet', 'csv')
    madis_hourly_research = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet', 'inclusive_csv')

    madis_daily_ = '/data/ssd2/madis/daily'

    solrad = os.path.join(d, 'dads', 'dem', 'rsun_tables', 'station_rsun')

    read_hourly_data(sites, madis_hourly_public, madis_daily_, solrad, shuffle=True, stype='dads',
                     bounds=(-180., 25., -60., 85.), overwrite=False, qaqc=True, plot=None, debug=True,
                     alt_src=madis_hourly_research, n_workers=8)

# ========================= EOF ====================================================================
