import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _process_file(args):
    file_path, test_year, years = args
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as e:
        station = os.path.splitext(os.path.basename(file_path))[0]
        rows = []
        for y in years:
            rows.append((station, y))
        return file_path, rows

    if df.empty:
        station = os.path.splitext(os.path.basename(file_path))[0]
        rows = []
        for y in years:
            rows.append((station, y))
        df.to_csv(file_path)
        return file_path, rows

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'repeated' in numeric_cols:
        numeric_cols.remove('repeated')

    if not numeric_cols:
        df['repeated'] = 0
        station = os.path.splitext(os.path.basename(file_path))[0]
        rows = []
        for y in years:
            sub = df[df.index.year == y]
            if sub.empty:
                rows.append((station, y))
        df.to_csv(file_path)
        return file_path, rows

    test = df[df.index.year == test_year]
    if test.empty:
        df['repeated'] = 0
        station = os.path.splitext(os.path.basename(file_path))[0]
        rows = []
        for y in years:
            sub = df[df.index.year == y]
            if sub.empty:
                rows.append((station, y))
        df.to_csv(file_path)
        return file_path, rows

    test_map = test[numeric_cols].copy()
    test_map['doy'] = test_map.index.dayofyear
    test_map = test_map.set_index('doy')

    doy = df.index.dayofyear
    test_aligned = test_map.reindex(doy)
    test_aligned.index = df.index

    r = df[numeric_cols].to_numpy()
    t = test_aligned[numeric_cols].to_numpy()
    eq = np.isclose(r, t, rtol=1e-10, atol=1e-12, equal_nan=True)
    repeated = eq.all(axis=1).astype(int)

    is_test_year = (df.index.year == test_year)
    repeated[is_test_year] = 0

    df['repeated'] = repeated
    station = os.path.splitext(os.path.basename(file_path))[0]
    rows = []
    for y in years:
        sub = df[df.index.year == y]
        if sub.empty:
            rows.append((station, y))
            continue
        if 'repeated' in sub.columns:
            if sub['repeated'].sum() == len(sub):
                rows.append((station, y))
    df.to_csv(file_path)
    return file_path, rows


def process_missing_landsat_data(table_dir, test_year=2023, num_workers=1, out_csv=None, years=None):
    files = [os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith('.csv')]
    if out_csv is not None:
        files = [p for p in files if os.path.basename(p) != os.path.basename(out_csv)]
    if years is None:
        years = list(range(1987, 2026))
    missing_rows = []
    if num_workers == 1:
        for f in tqdm(files):
            _, rows = _process_file((f, test_year, years))
            if rows:
                missing_rows.extend(rows)
    else:
        args = [(f, test_year, years) for f in files]
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for _, rows in tqdm(ex.map(_process_file, args), total=len(args)):
                if rows:
                    missing_rows.extend(rows)
    if out_csv is not None:
        if missing_rows:
            m = pd.DataFrame(missing_rows, columns=['station', 'year'])
            m.to_csv(out_csv, index=False)
        else:
            m = pd.DataFrame(columns=['station', 'year'])
            m.to_csv(out_csv, index=False)
    return files


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/nas'

    table_dir = os.path.join(d, 'dads', 'rs', 'landsat', 'station_data')
    test_year = 2023
    num_workers = 8
    out_missing = os.path.join(d, 'dads', 'rs', 'landsat', 'missing_station_years.csv')
    process_missing_landsat_data(table_dir, test_year=test_year, num_workers=num_workers, out_csv=out_missing,
                                 years=list(range(1987, 2026)))

# ========================= EOF ====================================================================
