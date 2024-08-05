import os
import glob

import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
import seaborn as sns
import matplotlib.pyplot as plt

from process.calc_eto import calc_asce_params
from process.station_parameters import station_par_map
from qaqc.qaqc_functions import rs_period_ratio_corr
from qaqc.calc_functions import calc_rs_tr, calc_rso


def read_hourly_data(stations, madis_src, madis_dst, shuffle=False, bounds=None, overwrite=False, plot=None):
    kw = station_par_map('dads')

    station_list = pd.read_csv(stations, index_col=kw['index'])
    station_list = station_list[station_list['source'] == 'madis']

    if shuffle:
        station_list = station_list.sample(frac=1)

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        # NLDAS-2 extent
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    record_ct, obs_ct = station_list.shape[0], 0
    for i, (fid, row) in enumerate(station_list.iterrows(), start=1):

        # if fid != 'DEEM8':
        #     continue

        lon, lat, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]
        print('{}: {} of {}; {:.2f}, {:.2f}'.format(fid, i, record_ct, lat, lon))

        out_file = os.path.join(madis_dst, '{}.csv'.format(fid))
        if os.path.exists(out_file) and not overwrite:
            print('{} exists, skipping'.format(fid))
            continue

        station_dir = os.path.join(madis_src, fid)
        files_ = [os.path.join(station_dir, f) for f in os.listdir(station_dir)]
        years = [int(f.split('.')[0].split('_')[-1]) for f in files_]

        df, first, local_tz = None, True, None
        for file_, yr in zip(files_, years):
            c = pd.read_csv(file_, index_col=0, parse_dates=True)
            c = c.sort_index()
            c['doy'] = c.index.dayofyear

            if first:
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lng=lon, lat=lat)
                local_tz = pytz.timezone(timezone_str)
                if timezone_str is None:
                    raise ValueError(f"Could not determine timezone for coordinates ({lat}, {lon})")

            c.index = c.index.tz_localize('GMT')
            c.index = c.index.tz_convert(local_tz)

            c['temperature'] -= 273.15
            es = 0.6108 * np.exp(17.27 * c['temperature'] / (c['temperature'] + 237.3))
            c['ea'] = (c['relHumidity'] / 100) * es
            c['rsds'] = c['solarRadiation'] * 0.0036

            c = process_daily_data(c, lat_=lat, elev_=elv, zw_=2.0)
            if c is None:
                continue

            if plot and c.shape[0] > 200:
                out_plot_file = os.path.join(plot, '{}_{}.png'.format(fid, yr))
                plot_daily_data(c, fid, yr, out_plot_file)

            if first:
                df = c.copy()
                first = False
            else:
                df = pd.concat([df, c], ignore_index=False, axis=0)

        df.to_csv(out_file)
        obs_ct += df.shape[0]
        print('wrote {}, {} records, {} total\n'.format(fid, df.shape[0], obs_ct))


def process_daily_data(hourly_df, lat_, elev_, zw_=2.0):
    """"""
    hourly_df['date'] = hourly_df.index.date
    hourly_df['precip_increment'] = hourly_df['precipAccum'].diff()
    hourly_df.loc[hourly_df['precip_increment'] < 0, 'precip_increment'] = 0
    hourly_df.loc[hourly_df['precip_increment'] > 50., 'precip_increment'] = 0

    valid_obs_count = hourly_df[['date']].groupby('date').agg({'date': 'count'}).copy()

    daily_df = hourly_df.groupby('date').agg(
        max_temp=('temperature', 'max'),
        min_temp=('temperature', 'min'),
        mean_temp=('temperature', 'mean'),
        prcp=('precip_increment', 'sum'),
        rsds=('rsds', 'sum'),
        ea=('ea', 'mean'),
        wind=('windSpeed', 'mean'),
        doy=('doy', 'first')
    ).copy()

    daily_df['obs_ct'] = valid_obs_count
    daily_df = daily_df[daily_df['obs_ct'] >= 18]
    daily_df.drop(columns=['obs_ct'], inplace=True)
    daily_df.index = pd.DatetimeIndex(daily_df.index)

    # asce_params = daily_df.parallel_apply(calc_asce_params, lat=lat_, elev=elev_, zw=10, axis=1)
    asce_params = daily_df.apply(calc_asce_params, lat=lat_, elev=elev_, zw=zw_, axis=1)

    try:
        daily_df[['mean_temp', 'vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                                         index=daily_df.index)
    except ValueError as e:
        print(e)
        return None

    return daily_df


def correct_data(meta, madis_daily_dir, madis_corrected, plot, target_sites):

    sites = pd.read_csv(meta, index_col='fid')

    files_ = list(os.listdir(madis_daily_dir))

    for f in files_:

        file_ = os.path.join(madis_daily_dir, f)
        site = f.split('.')[0]
        meta = sites.loc[site].to_dict()

        if target_sites:
            if site not in target_sites:
                continue

        df = pd.read_csv(file_, index_col=0, parse_dates=True)
        df['doy'] = df.index.dayofyear
        df['month'] = df.index.month
        rso = calc_rso(meta['latitude'], meta['elevation'], df['doy'], df['month'], df['ea'], df['rsds'])
        df['rso'] = rso[0] * 0.0864
        df.loc[df['rsds'] > (1.5 * df['rso']), 'rsds'] = np.nan
        rsds_corr = rs_period_ratio_corr(0, len(rso[0]), df['rsds'], rso[0], 100, 365)
        df['rsds_corr'] = rsds_corr[0]
        df = df.reindex(sorted(df.columns), axis=1)
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

    for i, var in enumerate(['vpd', 'rn', 'u2', 'mean_temp', 'eto'], start=1):
        sns.lineplot(data=pdf, x='doy', y=var, ax=axes[i])
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


def write_daily_maids_plots(madis_daily_dir, corrected, plot_dir, target_sites=None):
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

    sites = os.path.join(d, 'dads', 'met', 'stations', 'dads_stations.csv')
    madis_hourly = os.path.join(d, 'climate', 'madis', 'LDAD', 'mesonet', 'csv')
    madis_daily_ = os.path.join(d, 'dads', 'met', 'obs', 'madis')
    madis_daily_corr = os.path.join(d, 'dads', 'met', 'obs', 'madis_corrected')
    madis_plot_dir = os.path.join(d, 'dads', 'met', 'obs', 'plots', 'madis_{}')

    # read_hourly_data(sites, madis_in, madis_out, plot=None,
    #                  overwrite=True, shuffle=False, bounds=(-116., 45., -109., 49.))

    # read_hourly_data(sites, madis_in, madis_out, plot=None,
    #                  overwrite=False, shuffle=True, bounds=None)

    correct_data(sites, madis_daily_, madis_daily_corr, madis_plot_dir, target_sites=['PNTM8'])
# ========================= EOF ====================================================================
