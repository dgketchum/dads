import requests

import pandas as pd
import io


def get_madis():

    base_url = "https://madis-data.ncep.noaa.gov/madis/v1/"
    endpoint = "metar/netcdf"
    params = {
        "startDate": "20240724",
        "endDate": "20240725",
        "station": "KBZN"  # Example station ID
    }

    response = requests.get(base_url + endpoint, params=params)
    response.raise_for_status()  # Raise an error if request fails

    # Save the downloaded NetCDF data
    with open("metar_data.nc", "wb") as file:
        file.write(response.content)



def download_raws_data(state, year, month):
    """Downloads RAWS data for a given state, year, and month."""

    base_url = "https://fam.nwcg.gov/fam-web/weather/raws/"
    file_name = f"{state.upper()}{year}{month:02d}.csv"  # Format filename
    url = base_url + file_name

    response = requests.get(url)

    if response.status_code == 200:
        data = response.content.decode("utf-8")  # Decode the content
        df = pd.read_csv(io.StringIO(data), skiprows=1)  # Parse as CSV (skip header)
        return df
    else:
        print(f"Error downloading data for {state}, {year}-{month:02d}: {response.status_code}")
        return None


def get_ncei():
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"  # Data Service API
    dataset = "global-hourly"
    stations = "USW00023129"  # Chicago Midway Airport
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data_types = "TMP,DEWP,SLP"  # Temperature, dew point, sea level pressure

    params = {
        "dataset": dataset,
        "stations": stations,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": data_types,
        "format": "json",  # Request JSON format
        # Add other parameters as needed (e.g., units)
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Process the retrieved data
    else:
        print("Error:", response.status_code)


def get_acis():
    base_url = "https://data.rcc-acis.org/"
    endpoint = "StnData"  # Choose the appropriate API call
    params = {
        "sid": "KMSO",  # Station ID (Missoula International Airport)
        "sdate": "2023-07-01",  # Start date
        "edate": "2023-07-31",  # End date
        "elems": [
            {"name": "maxt"},  # Maximum temperature
            {"name": "mint"},  # Minimum temperature
            {"name": "pcpn"}  # Precipitation
        ],
    }

    response = requests.post(base_url + endpoint, json={"params": params})

    if response.status_code == 200:
        data = response.json()
        # Process the retrieved data
    else:
        print("Error:", response.status_code)


if __name__ == '__main__':
    st = 'MT'
    yr = 2020
    m = 7
    download_raws_data(st, yr, m)

# ========================= EOF ====================================================================
