import os
import json
import glob
from datetime import datetime
import collections
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pprint import pprint

import numpy as np
import pandas as pd

from utils.station_parameters import station_par_map


def check_madis_files(subdir_arg, output_file_prefix_arg, root_dir_arg,
                      secondary_subdir_name_check_arg=None):
    CUTOFF_DATE_STR = '2024-04-01'
    MODIFIED_COUNT_THRESHOLD = 1000

    BASE_DIR = os.path.join(root_dir_arg, subdir_arg)
    text_output_filename = f'{output_file_prefix_arg}_modification_summary.txt'
    plot_output_filename = f'{output_file_prefix_arg}_madis_hist.png'

    found_non_empty_secondary_subdir_overall = False

    start_date_str_arg = '2001-01-01'
    end_date_str_arg = '2025-05-15'

    try:
        start_date_obj = pd.to_datetime(start_date_str_arg)
        end_date_obj = pd.to_datetime(end_date_str_arg)
        if end_date_obj < start_date_obj:
            print(f"Error: end_date_str ({end_date_str_arg}) cannot be before start_date_str ({start_date_str_arg}).")
            return False
    except ValueError as e:
        print(f"Error: Invalid date string format for start_date_str or end_date_str: {e}. Use YYYY-MM-DD.")
        return False

    try:
        cutoff_dt = datetime.datetime.strptime(CUTOFF_DATE_STR, '%Y-%m-%d')
        cutoff_timestamp = cutoff_dt.timestamp()
        print(f"Checking for files modified after: {cutoff_dt} (Timestamp: {cutoff_timestamp})")
    except ValueError:
        print(f"Error: Invalid CUTOFF_DATE_STR format in script: '{CUTOFF_DATE_STR}'. Use YYYY-MM-DD.")
        return False

    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory not found: {BASE_DIR}")
        return False

    modified_files_by_month = collections.defaultdict(list)
    processed_dirs = 0
    processed_files = 0

    print(f"Starting file scan in: {BASE_DIR}")

    station_dirs_list = []
    try:
        station_dirs_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
        print(f"Found {len(station_dirs_list)} potential station directories in {BASE_DIR}.")
    except OSError as e:
        print(f"Error listing contents of base directory {BASE_DIR}: {e}")
        return False

    total_station_dirs_to_scan = len(station_dirs_list)

    for i, dir_name in enumerate(station_dirs_list):
        processed_dirs += 1
        if processed_dirs > 0 and processed_dirs % 1000 == 0:
            print(f"... scanned {processed_dirs}/{total_station_dirs_to_scan} station directories ...")

        current_station_dir_path = os.path.join(BASE_DIR, dir_name)

        if secondary_subdir_name_check_arg and secondary_subdir_name_check_arg.strip():
            path_to_secondary_subdir = os.path.join(current_station_dir_path, secondary_subdir_name_check_arg)
            if os.path.isdir(path_to_secondary_subdir):
                try:
                    if os.listdir(path_to_secondary_subdir):
                        found_non_empty_secondary_subdir_overall = True
                except OSError as e:
                    print(
                        f"Warning: Could not list contents of secondary subdir {path_to_secondary_subdir} for station {dir_name}: {e}")

        try:
            files_in_station_dir = os.listdir(current_station_dir_path)
        except OSError as e:
            print(f"Warning: Could not list files in {current_station_dir_path}: {e}")
            continue

        for filename in files_in_station_dir:
            if filename.endswith('.csv') and '_' in filename:
                processed_files += 1
                current_file_full_path = os.path.join(current_station_dir_path, filename)
                try:
                    parts = filename.split('_')
                    if len(parts) < 2:
                        continue
                    year_month_str = parts[-1][:6]

                    if len(year_month_str) == 6 and year_month_str.isdigit():
                        mod_timestamp = os.path.getmtime(current_file_full_path)
                        if mod_timestamp > cutoff_timestamp:
                            modified_files_by_month[year_month_str].append(current_file_full_path)
                except FileNotFoundError:
                    print(f"Warning: File not found during stat: {current_file_full_path}")
                except Exception as e:
                    print(f"Warning: Error processing file {current_file_full_path}: {e}")

    print(f"\nScan complete. Checked {processed_files} files across {processed_dirs} directories.")

    print(f"\nWriting summary to: {text_output_filename}")
    summary_data = []
    sorted_months = sorted(modified_files_by_month.keys())

    for month_str in sorted_months:
        file_list = modified_files_by_month[month_str]
        count = len(file_list)
        if count > 0:
            file_list.sort()
            first_file = os.path.basename(file_list[0])
            last_file = os.path.basename(file_list[-1])
            summary_data.append({
                "Month": month_str,
                "Count": count,
                "FirstFile": first_file,
                "LastFile": last_file
            })

    try:
        with open(text_output_filename, 'w') as f:
            f.write(f"Summary of CSV Files Modified After {CUTOFF_DATE_STR}\n")
            f.write(
                f"Based on scan of {processed_dirs} director{'y' if processed_dirs == 1 else 'ies'} found within {BASE_DIR}.\n")
            f.write("Generated on: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("-" * 80 + "\n")
            f.write("{:<10s} {:<10s} {:<30s} {:<30s}\n".format("Month", "Count", "First Modified (Alphabetical)",
                                                               "Last Modified (Alphabetical)"))
            f.write("-" * 80 + "\n")
            if summary_data:
                for item in summary_data:
                    f.write("{:<10s} {:<10d} {:<30s} {:<30s}\n".format(
                        item["Month"], item["Count"], item["FirstFile"], item["LastFile"]
                    ))
            else:
                f.write("No files found modified after the cutoff date within the scanned directories.\n")
        print("Summary file written successfully.")
    except IOError as e:
        print(f"Error writing summary file {text_output_filename}: {e}")

    hist_plot_start_date = start_date_obj.replace(day=1)
    hist_plot_end_date = end_date_obj.replace(day=1)

    date_index = pd.date_range(
        start=hist_plot_start_date,
        end=hist_plot_end_date,
        freq='MS'
    )

    all_months_counts = pd.Series(0, index=date_index)

    for ym_str, files in modified_files_by_month.items():
        try:
            month_dt = pd.to_datetime(ym_str, format='%Y%m')
            if month_dt in all_months_counts.index:
                all_months_counts[month_dt] = len(files)
        except ValueError:
            print(f"Warning: Could not parse date from count key: {ym_str}")

    print(f"\n--- Generating Time Tuples for Months with < {MODIFIED_COUNT_THRESHOLD} Recently Modified Files ---")
    incomplete_month_tuples = []

    for month_start_dt_in_range in all_months_counts.index:
        year = month_start_dt_in_range.year
        month = month_start_dt_in_range.month
        year_month_str_key = month_start_dt_in_range.strftime('%Y%m')

        modified_count_for_month = len(modified_files_by_month.get(year_month_str_key, []))

        if modified_count_for_month < MODIFIED_COUNT_THRESHOLD:
            last_day_of_month = month_start_dt_in_range.days_in_month
            tuple_start_time_str = f"{year}{month:02d}01 00"
            tuple_end_time_str = f"{year}{month:02d}{last_day_of_month:02d} 23"
            incomplete_month_tuples.append((tuple_start_time_str, tuple_end_time_str))

    print(
        f"Found {len(incomplete_month_tuples)} months potentially needing processing (Count < {MODIFIED_COUNT_THRESHOLD} based on scan).")
    print("Processing Time Tuples:")
    if incomplete_month_tuples:
        print(incomplete_month_tuples)
    else:
        print("No months found with fewer than {} recently modified files within the scanned data.".format(
            MODIFIED_COUNT_THRESHOLD))

    print("\n--- Files Modified After {} By Month (YYYYMM) (Based on Scan) ---".format(CUTOFF_DATE_STR))
    counts_df = all_months_counts.reset_index()
    counts_df.columns = ['MonthDateTime', 'ModifiedFileCount']
    counts_df['MonthStr'] = counts_df['MonthDateTime'].dt.strftime('%Y%m')

    print("Months with recently modified files found in scan (Count):")
    non_zero_counts_df = counts_df[counts_df['ModifiedFileCount'] > 0]
    if not non_zero_counts_df.empty:
        print(non_zero_counts_df[['MonthStr', 'ModifiedFileCount']].to_string(index=False))
    else:
        print("No files found modified after the cutoff date within the scanned directories.")

    print("\nGenerating histogram...")
    fig, ax = plt.subplots(figsize=(18, 7))
    all_months_counts.plot(kind='bar', ax=ax, width=0.8)

    title_start_month_str = date_index.min().strftime('%Y-%m')
    title_end_month_str = date_index.max().strftime('%Y-%m')

    title_main_part = f'CSV Files Modified After {CUTOFF_DATE_STR}\n(Counts per Month, {title_start_month_str} to {title_end_month_str}'

    details_suffix = ""
    path_descriptor = f'"{subdir_arg}"' if subdir_arg and subdir_arg.strip() else f'root path ("{root_dir_arg}")'

    if total_station_dirs_to_scan > 0:
        details_suffix = f' - Based on scan of {processed_dirs} director{"y" if processed_dirs == 1 else "ies"} in {path_descriptor}'
    else:
        details_suffix = f' - No station directories found to scan in {path_descriptor}'

    title = f'{title_main_part}{details_suffix})'
    ax.set_title(title)

    ax.set_ylabel('Number of Modified Files')
    ax.set_xlabel('Month (YYYY-MM)')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))

    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(plot_output_filename)
        print(f"Histogram saved to {plot_output_filename}")
    except Exception as e:
        print(f"Error saving histogram: {e}")

    plt.show()
    plt.close(fig)

    # Report status of the secondary subdirectory check
    if secondary_subdir_name_check_arg and secondary_subdir_name_check_arg.strip():
        if found_non_empty_secondary_subdir_overall:
            print(
                f"\nSecondary Directory Check: At least one non-empty instance of '{secondary_subdir_name_check_arg}' was found within the processed station directories.")
        else:
            print(
                f"\nSecondary Directory Check: No non-empty instances of '{secondary_subdir_name_check_arg}' were found within the processed station directories.")

    print("Script finished.")
    return True


def check_madis_index(stations, csv_hourly, parquet_daily, source='madis', out_json=None):
    kw = station_par_map(source)
    stations = pd.read_csv(stations, index_col=kw['index'])
    stations.sort_index(inplace=True)

    ct, dupes = 0, 0

    for i, (station_id, row) in enumerate(stations.iterrows(), start=1):
        ct += 1

        sta_file = os.path.join(parquet_daily, f'{station_id}.parquet')

        if not os.path.exists(sta_file):
            continue

        sdf = pd.read_parquet(sta_file)

        obs_cols = [f'{c}_obs' for c in sdf.columns]
        sdf.columns = obs_cols
        bad_csv = {}

        if sdf.index.has_duplicates:
            dupes += 1

            station_csv_dir = os.path.join(csv_hourly, station_id)
            if not os.path.isdir(station_csv_dir):
                print(f"    WARNING: Source CSV directory not found: {station_csv_dir}")
                continue

            csv_files = glob.glob(os.path.join(station_csv_dir, '*.csv'))

            if not csv_files:
                print(f"    INFO: No source CSV files found for station {station_id}.")

            for csv_path in sorted(csv_files):

                filename = os.path.basename(csv_path)

                filename_yyyymm = filename.split('_')[1].split('.')[0]
                assert (len(filename_yyyymm) == 6 and filename_yyyymm.isdigit())

                monthly_df = pd.read_csv(csv_path, index_col=0, parse_dates=True, on_bad_lines='skip')
                if monthly_df.empty:
                    continue

                data_yyyymm = list(set(monthly_df.index.strftime('%Y%m')))

                if len(data_yyyymm) == 0:
                    # print(f'empty {os.path.basename(filename)}')
                    mismatches_found = False

                elif len(data_yyyymm) > 1:
                    # print(f'multiple months in {os.path.basename(filename)}: {data_yyyymm}')
                    mismatches_found = True

                elif data_yyyymm[0] != filename_yyyymm:
                    # print(f'mismatch in {os.path.basename(filename)}: {data_yyyymm}')
                    mismatches_found = True

                else:
                    continue

                if mismatches_found:
                    if station_id not in bad_csv:
                        bad_csv[station_id] = [filename]
                    else:
                        bad_csv[station_id].append(filename)

            print(f'{station_id} has {len(bad_csv[station_id])} mismatched files, {dupes} of {ct}')

        with open(out_json, 'w') as fp:
            json.dump(bad_csv, fp, indent=4)



if __name__ == "__main__":

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    madis = os.path.join(d, 'climate', 'madis')

    mesonet_dir_public = os.path.join(madis, 'LDAD_public', 'mesonet')
    out_dir_public = os.path.join(mesonet_dir_public, 'inclusive_csv')

    mesonet_dir_research = os.path.join(madis, 'LDAD', 'mesonet')
    out_dir_research = os.path.join(mesonet_dir_research, 'inclusive_csv')

    # check_madis_files(out_dir_public, 'madis_check_15MAY2025', madis)

    data = '/data/ssd2'
    sites = os.path.join(d, 'dads', 'met', 'stations', 'madis_17MAY2025_mgrs.csv')
    pqt = os.path.join(data, 'madis', 'daily')
    out_json_ = os.path.join(data, 'madis', 'bad_files_12JUN2025.json')

    check_madis_index(sites, out_dir_research, pqt, source='madis', out_json=out_json_)
# ========================= EOF ====================================================================
