"""Build a stationId → wind_sensor_ht crosswalk from Synoptic API metadata.

MADIS reports all MesoWest-ingested stations under ``dataProvider=MesoWest``,
hiding the true originating network and sensor height.  Synoptic knows each
station's real network (MNET_ID) and, for some stations, the explicit wind
sensor height (``SENSOR_VARIABLES.wind_speed.position``).

Three-tier height assignment:
  1. Explicit sensor height from Synoptic (when available)
  2. Network default from a curated table (AGRIMET→3m, PGN→6m, etc.)
  3. Fallback 10.0 m

Output
------
``artifacts/synoptic_wind_height_crosswalk.csv`` with columns:
  stationId, mnet_id, network_name, sensor_ht_explicit, wind_sensor_ht
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import pandas as pd

# Curated network → default wind sensor height (metres).
# Sources: GSI urma2p5_provider_windheight, network documentation,
# confirmed against Synoptic explicit sensor positions where available.
NETWORK_DEFAULT_HT: dict[int, float] = {
    # Confirmed against Synoptic explicit sensor positions
    11: 3.0,  # AGRIMET — all 70 explicit are 3.0 m
    3002: 3.0,  # MT-MESO — all 90 explicit are 3.0 m
    2: 6.1,  # RAWS — 450/503 explicit are 6.1 m (GSI says 6.0)
    # From GSI urma2p5_provider_windheight
    65: 2.0,  # APRSWXNET/CWOP — GSI APRSWXNE = 2.0 m
    221: 2.0,  # CoAgMet — GSI CoAgMet = 2.0 m
    143: 3.0,  # DEOS — GSI DEOS = 3.0 m
    3008: 3.0,  # NDAWN — GSI NDAWN = 3.0 m
    3007: 2.0,  # WSU-AGNET — ag weather, 2 m standard
    114: 2.0,  # PAWS/AgWeatherNet — ag weather, 2 m standard
    26: 3.0,  # UTAH CLIMATE CENTER — GSI UTAH CLI = 3.0 m
    265: 6.0,  # PGN (Portland General Electric) — GSI PGE = 6.0 m
    287: 6.0,  # PSE (Puget Sound Energy) — utility, similar to PGE
    29: 2.0,  # SCAN — USDA soil/climate, 2 m standard
    194: 2.0,  # UCC-AGNET — ag weather
}

FALLBACK_HT = 10.0

AUTH_URL = "https://api.synopticdata.com/auth/v2/"
META_URL = "https://api.synopticdata.com/v2/stations/metadata"
NETS_URL = "https://api.synopticdata.com/v2/networks"


def _get_token(api_key: str) -> str:
    """Exchange an API key for a public token."""
    req = urllib.request.Request(
        AUTH_URL, headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["TOKEN"]


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def _fetch_network_names(token: str) -> dict[int, str]:
    """Return {mnet_id: shortname} for all Synoptic networks."""
    data = _fetch_json(f"{NETS_URL}?token={token}")
    return {int(n["ID"]): n["SHORTNAME"] for n in data["MNET"]}


def _fetch_station_metadata(
    token: str, bbox: str, status: str = "active"
) -> list[dict]:
    """Fetch metadata for all stations with wind_speed in a bounding box."""
    url = (
        f"{META_URL}?token={token}"
        f"&bbox={bbox}"
        f"&sensorvars=1"
        f"&vars=wind_speed"
        f"&status={status}"
    )
    data = _fetch_json(url)
    summary = data.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        raise RuntimeError(f"Synoptic API error: {summary}")
    return data["STATION"]


def build_crosswalk(
    api_key_path: str,
    bbox: str = "-125,42,-104,49",
    out_path: str = "artifacts/synoptic_wind_height_crosswalk.csv",
) -> pd.DataFrame:
    """Query Synoptic and build stationId → wind_sensor_ht crosswalk."""
    api_key = open(api_key_path).read().strip()
    token = _get_token(api_key)
    print("Synoptic token obtained", flush=True)

    net_names = _fetch_network_names(token)
    print(f"Loaded {len(net_names)} network definitions", flush=True)

    stations = _fetch_station_metadata(token, bbox)
    print(f"Fetched metadata for {len(stations)} stations with wind_speed", flush=True)

    rows = []
    for stn in stations:
        stid = stn["STID"]
        mnet_id = int(stn["MNET_ID"])
        net_name = net_names.get(mnet_id, "UNKNOWN")

        # Extract explicit wind sensor height
        svars = stn.get("SENSOR_VARIABLES", {})
        ws = svars.get("wind_speed", {})
        sensor_ht_explicit = None
        if ws:
            first = list(ws.values())[0]
            pos = first.get("position")
            if pos is not None:
                sensor_ht_explicit = float(pos)

        # Three-tier assignment
        if sensor_ht_explicit is not None:
            wind_sensor_ht = sensor_ht_explicit
        elif mnet_id in NETWORK_DEFAULT_HT:
            wind_sensor_ht = NETWORK_DEFAULT_HT[mnet_id]
        else:
            wind_sensor_ht = FALLBACK_HT

        rows.append(
            {
                "stationId": stid,
                "mnet_id": mnet_id,
                "network_name": net_name,
                "sensor_ht_explicit": sensor_ht_explicit,
                "wind_sensor_ht": wind_sensor_ht,
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary
    non_default = df[df["wind_sensor_ht"] != FALLBACK_HT]
    print(f"\nCrosswalk: {len(df)} stations total", flush=True)
    print(f"  with non-10m height: {len(non_default)}", flush=True)
    print("  height distribution:", flush=True)
    for ht, ct in df["wind_sensor_ht"].value_counts().sort_index().items():
        print(f"    {ht:5.1f} m: {ct}", flush=True)
    print(f"\nWritten to {out_path}", flush=True)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Synoptic wind height crosswalk")
    p.add_argument(
        "--api-key",
        default=os.path.expanduser("~/synoptic_key.txt"),
        help="Path to file containing Synoptic API key.",
    )
    p.add_argument(
        "--bbox",
        default="-125,42,-104,49",
        help="Bounding box: west,south,east,north.",
    )
    p.add_argument(
        "--out",
        default="artifacts/synoptic_wind_height_crosswalk.csv",
        help="Output CSV path.",
    )
    args = p.parse_args()
    build_crosswalk(api_key_path=args.api_key, bbox=args.bbox, out_path=args.out)
