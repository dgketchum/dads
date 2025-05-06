import os
import datetime
import collections
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd  # Using pandas simplifies date range generation and plotting

# import calendar # Only needed if using calendar.monthrange instead of pandas days_in_month

# --- Configuration ---
BASE_DIR = '/home/dgketchum/data/IrrigationGIS/climate/madis/LDAD/mesonet/inclusive_csv/'
CUTOFF_DATE_STR = '2024-04-01'
START_YEAR = 2001
END_YEAR = 2025
END_MONTH = 4  # Inclusive (up to April 2025)
OUTPUT_TEXT_FILE = 'modification_summary.txt'  # Name for the output text file
MODIFIED_COUNT_THRESHOLD = 1000  # Threshold for generating processing tuples

# --- Preparation ---
try:
    # Convert cutoff date string to a datetime object, then to a timestamp
    # Using timezone-aware comparison might be more robust if system/file times differ
    cutoff_dt = datetime.datetime.strptime(CUTOFF_DATE_STR, '%Y-%m-%d')
    cutoff_timestamp = cutoff_dt.timestamp()
    print(f"Checking for files modified after: {cutoff_dt} (Timestamp: {cutoff_timestamp})")

except ValueError:
    print(f"Error: Invalid CUTOFF_DATE_STR format: '{CUTOFF_DATE_STR}'. Use YYYY-MM-DD.")
    exit()

if not os.path.isdir(BASE_DIR):
    print(f"Error: Base directory not found: {BASE_DIR}")
    exit()

# Use defaultdict to store lists of modified files per month
modified_files_by_month = collections.defaultdict(list)
processed_dirs = 0
processed_files = 0

print(f"Starting file scan in: {BASE_DIR}")

# --- File System Traversal and Data Collection (User's Modified Version) ---
try:
    # List only directories directly under BASE_DIR
    dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    print(f"Found {len(dirs)} potential station directories.")
except OSError as e:
    print(f"Error listing base directory {BASE_DIR}: {e}")
    exit()

for i, dir_ in enumerate(dirs):
    processed_dirs += 1
    if processed_dirs % 1000 == 0 and processed_dirs > 0:
        print(f"... scanned {processed_dirs}/{len(dirs)} station directories ...")

    current_dir_path = os.path.join(BASE_DIR, dir_)
    try:
        files = os.listdir(current_dir_path)
    except OSError as e:
        print(f"Warning: Could not list files in {current_dir_path}: {e}")
        continue  # Skip this directory

    for filename in files:
        if filename.endswith('.csv') and '_' in filename:
            processed_files += 1
            try:
                # Extract YYYYMM from filename like 'AP156_200106.csv'
                parts = filename.split('_')
                if len(parts) < 2: continue  # Skip if no underscore
                year_month_str = parts[-1][:6]  # Get '200106' part

                # Validate format
                if len(year_month_str) == 6 and year_month_str.isdigit():
                    file_path = os.path.join(current_dir_path, filename)

                    # Get modification time
                    mod_timestamp = os.path.getmtime(file_path)

                    # Compare modification time
                    if mod_timestamp > cutoff_timestamp:
                        # Store the full path of the modified file
                        modified_files_by_month[year_month_str].append(file_path)
                else:
                    # Optional: Log filenames that don't match the expected pattern
                    # print(f"Skipping file with unexpected name format: {filename} in {dir_}")
                    pass

            except FileNotFoundError:
                # This might happen in race conditions, less likely with listdir -> getmtime
                print(f"Warning: File not found during stat: {file_path}")
            except Exception as e:
                print(f"Warning: Error processing file {os.path.join(current_dir_path, filename)}: {e}")

    # User's break condition
    # if processed_dirs >= 5000:
    #     print(f"\nStopping scan after {processed_dirs} directories due to limit.")
    #     break

print(f"\nScan complete. Checked {processed_files} files across {processed_dirs} directories.")

# --- Process Collected Data and Write Text File ---
print(f"\nWriting summary to: {OUTPUT_TEXT_FILE}")

summary_data = []
# Sort months chronologically for processing and output
sorted_months = sorted(modified_files_by_month.keys())

for month_str in sorted_months:
    file_list = modified_files_by_month[month_str]
    count = len(file_list)
    if count > 0:
        file_list.sort()  # Sort alphabetically by full path
        first_file = os.path.basename(file_list[0])  # Get basename for report
        last_file = os.path.basename(file_list[-1])  # Get basename for report
        summary_data.append({
            "Month": month_str,
            "Count": count,
            "FirstFile": first_file,
            "LastFile": last_file
        })

# Write the summary text file
try:
    with open(OUTPUT_TEXT_FILE, 'w') as f:
        f.write(f"Summary of CSV Files Modified After {CUTOFF_DATE_STR}\n")
        # Note: Summary reflects only the directories scanned if break condition was met
        f.write(f"Based on scan of {processed_dirs} directories.\n")
        f.write("Generated on: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("-" * 80 + "\n")
        # Header - adjust padding as needed
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
    print(f"Error writing summary file: {e}")

# --- Prepare Data for Histogram (using counts derived from collected files) ---

# Generate all months in the range 2001-01 to 2025-04
date_index = pd.date_range(
    start=f'{START_YEAR}-01-01',
    end=f'{END_YEAR}-{END_MONTH:02d}-01',  # Generate up to the first day of the target end month
    freq='MS'  # Month Start frequency
)

# Create a series with all months initialized to 0 counts
all_months_counts = pd.Series(0, index=date_index)

# Populate the series with actual counts derived from the collected file lists
# These counts reflect only the scanned directories if the loop broke early
for ym_str, files in modified_files_by_month.items():
    try:
        # Convert YYYYMM string to a datetime object for indexing
        month_dt = pd.to_datetime(ym_str, format='%Y%m')
        if month_dt in all_months_counts.index:
            all_months_counts[month_dt] = len(files)  # Get count from list length
    except ValueError:
        print(f"Warning: Could not parse date from count key: {ym_str}")

# --- Generate List of Tuples for Incomplete Months --- <<< ADDED LOGIC HERE
print(f"\n--- Generating Time Tuples for Months with < {MODIFIED_COUNT_THRESHOLD} Recently Modified Files ---")
incomplete_month_tuples = []

# Iterate through all months in the defined range using the pandas date_index
# This considers the full date range, regardless of scan limit or actual files found
for month_start_dt in all_months_counts.index:
    year = month_start_dt.year
    month = month_start_dt.month
    year_month_str = month_start_dt.strftime('%Y%m')  # Key for our dictionary

    # Get the count of recently modified files *found in the potentially limited scan*
    modified_count = len(modified_files_by_month.get(year_month_str, []))

    # Check if the count is below the threshold
    if modified_count < MODIFIED_COUNT_THRESHOLD:
        # Calculate the last day of the month
        last_day = month_start_dt.days_in_month  # using pandas Timestamp property

        # Format the start and end time strings
        start_time_str = f"{year}{month:02d}01 00"
        end_time_str = f"{year}{month:02d}{last_day:02d} 23"

        # Add the tuple to the list
        incomplete_month_tuples.append((start_time_str, end_time_str))

# Print the generated list to the terminal
print(
    f"Found {len(incomplete_month_tuples)} months potentially needing processing (Count < {MODIFIED_COUNT_THRESHOLD} based on scan).")
print("Processing Time Tuples:")
if incomplete_month_tuples:
    print(incomplete_month_tuples)
else:
    print("No months found with fewer than {} recently modified files within the scanned data.".format(
        MODIFIED_COUNT_THRESHOLD))
# --- END OF ADDED LOGIC ---


# --- Output Results to Console (Counts Summary) ---
print("\n--- Files Modified After {} By Month (YYYYMM) (Based on Scan) ---".format(CUTOFF_DATE_STR))
# Create a DataFrame for easier viewing/sorting if needed
counts_df = all_months_counts.reset_index()
counts_df.columns = ['Month', 'ModifiedFileCount']
counts_df['MonthStr'] = counts_df['Month'].dt.strftime('%Y%m')

# Print non-zero counts
print("Months with recently modified files found in scan (Count):")
non_zero_counts = counts_df[counts_df['ModifiedFileCount'] > 0]
if not non_zero_counts.empty:
    print(non_zero_counts[['MonthStr', 'ModifiedFileCount']].to_string(index=False))
else:
    print("No files found modified after the cutoff date within the scanned directories.")

# --- Create Histogram ---
print("\nGenerating histogram...")

fig, ax = plt.subplots(figsize=(18, 7))  # Wider figure for readability

# Plotting the series directly
all_months_counts.plot(kind='bar', ax=ax, width=0.8)

# Update title to reflect potential partial scan
title = f'CSV Files Modified After {CUTOFF_DATE_STR}\n'
title += f'(Counts per Month, {START_YEAR}-01 to {END_YEAR}-{END_MONTH:02d}'
if processed_dirs < len(dirs):  # Check if scan was stopped early
    title += f' - Based on partial scan of {processed_dirs}/{len(dirs)} dirs'
title += ')'
ax.set_title(title)

ax.set_ylabel('Number of Modified Files')
ax.set_xlabel('Month (YYYY-MM)')

# Improve x-axis labels
ax.xaxis.set_major_locator(mdates.YearLocator())  # Tick every year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Minor ticks Jan/Jul

plt.xticks(rotation=90)  # Rotate labels for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent labels overlapping

# Save the figure BEFORE showing it
try:
    plt.savefig('madis_hist_04MAY.png')
    print("Histogram saved to maids_hist.png")
except Exception as e:
    print(f"Error saving histogram: {e}")

plt.show()

print("Script finished.")
