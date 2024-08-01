import os

import pandas as pd


def proc_modis(extracts, output):

    filenames = [f for f in os.listdir(extracts)]
    dates = ['-'.join(f.replace('.csv', '').split("_")[3:6]) for f in filenames]
    dts = []
    for d in dates:
        try:
            dts.append(pd.to_datetime(d))
        except Exception as e:
            print(e, d)

    index = pd.DatetimeIndex(dts)
    index.sort()

    df = pd.DataFrame()
    for filename in os.listdir(extracts):
        if filename.startswith("dads_stations_") and filename.endswith(".csv"):
            filepath = os.path.join(extracts, filename)
            df = pd.read_csv(filepath)
            df.set_index('fid', inplace=True)
            df = df.T

            date_str = filename.replace('.csv', '')
            date_str = date_str.split("_")[3:6]
            date_str = "-".join(date_str)
            df.index = pd.to_datetime(date_str)
            df.index.name = 'date'

            # Write each fid to separate csv
            for fid in df.columns:
                fid_df = df[[fid]].dropna()
                fid_filename = f"dads_stations_fid_{fid}_{date_str}.csv"
                fid_filepath = os.path.join(output, fid_filename)
                fid_df.to_csv(fid_filepath)



if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/dads'
    if not os.path.exists(d):
        d = d = '/home/dgketchum/data/IrrigationGIS/dads'

    daily = os.path.join(d, 'rs', 'dads_stations', 'modis', 'extracts')
    formed = os.path.join(d, 'rs', 'dads_stations', 'modis', 'prepped')
    proc_modis(daily, formed)

# ========================= EOF ================================================================================
