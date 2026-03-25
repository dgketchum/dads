"""Adapter between obsmet releases and dads prep scripts.

Reads QC-passed daily station observations from an obsmet channel/release
and returns DataFrames in the conventions expected by dads builders.

Two access modes
----------------
channel mode (production)
    Uses the /nas/climate/obsmet/channels/{channel} symlink created by
    obsmet's build_release().  Pass channel="prod" (default).

direct-path mode (development)
    Pass por_path="/path/to/station_por/madis/permissive" to read straight
    from a permissive product directory, bypassing the channels mechanism.
    Use this until build_release() populates the prod channel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from obsmet.products.release import CHANNELS_ROOT

# dads name → obsmet column name (when they differ)
_DADS_TO_OBSMET = {
    "ea": "ea_compiled",
}


def resolve_release(channel: str = "prod") -> Path:
    """Resolve a channel symlink to its concrete release directory."""
    link = CHANNELS_ROOT / channel
    if not link.exists():
        raise FileNotFoundError(f"Channel not found: {link}")
    return link.resolve()


def load_station_daily(
    source: str,
    *,
    channel: str = "prod",
    por_path: str | Path | None = None,
    variables: list[str] | None = None,
    fids: set[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load QC-passed daily obs from an obsmet release.

    Parameters
    ----------
    source : str
        obsmet source identifier (e.g. "madis", "ghcnd").
    channel : str
        Release channel (default "prod").  Ignored when por_path is given.
    por_path : str | Path | None
        Direct path to a station_por leaf directory (e.g. the permissive
        product dir).  When provided, bypasses the channels/release
        infrastructure.  Production should use build_release() instead.
    variables : list[str] | None
        dads variable names to keep (e.g. ["tmax", "tmin", "ea"]).
        If None, keep all metric columns.
    fids : set[str] | None
        Station IDs to include.  If None, load all stations.
    start, end : str | None
        Date bounds (inclusive), e.g. "2018-01-01".

    Returns
    -------
    DataFrame indexed by (fid, day) with requested variable columns.
    """
    # Build the obsmet column set to read
    obsmet_cols: set[str] | None = None
    if variables is not None:
        obsmet_cols = {_DADS_TO_OBSMET.get(v, v) for v in variables}

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    if por_path is not None:
        # NOTE: Direct path bypasses channels/release infrastructure.
        # Production should use build_release() which sets up the
        # /nas/climate/obsmet/channels/{channel} symlink.
        por_dir = Path(por_path)
        manifest = pd.read_parquet(por_dir / "manifest.parquet")
        manifest = manifest.loc[manifest["source"] == source]
        manifest = manifest.loc[manifest["state"] == "done"]
        station_iter = [
            (
                row["key"].split(":", 1)[1],
                por_dir / (row["key"].replace(":", "_") + ".parquet"),
            )
            for _, row in manifest.iterrows()
        ]
    else:
        release_dir = resolve_release(channel)
        por_dir = release_dir / "station_por" / source
        if not por_dir.is_dir():
            raise FileNotFoundError(f"No station_por for source={source}: {por_dir}")
        station_iter = []
        for pq_path in sorted(por_dir.glob("*.parquet")):
            stem = pq_path.stem
            fid = stem.split(":", 1)[1] if ":" in stem else stem
            station_iter.append((fid, pq_path))

    chunks: list[pd.DataFrame] = []
    for fid, pq_path in station_iter:
        if fids is not None and fid not in fids:
            continue

        df = pd.read_parquet(pq_path)

        # QC filter: keep pass + suspect, drop fail only.
        # "suspect" rows are typically flagged for missing ancillary variables
        # (e.g. dd_missing) but have valid target obs.
        if "qc_state" in df.columns:
            df = df.loc[df["qc_state"].isin(("pass", "suspect"))]
        if df.empty:
            continue

        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if start_ts is not None:
                df = df.loc[df["date"] >= start_ts]
            if end_ts is not None:
                df = df.loc[df["date"] <= end_ts]
        if df.empty:
            continue

        # Variable selection
        if obsmet_cols is not None:
            # ea_compiled fallback: if ea_compiled missing but ea present, use ea
            effective = set()
            for c in obsmet_cols:
                if c in df.columns:
                    effective.add(c)
                elif c == "ea_compiled" and "ea" in df.columns:
                    effective.add("ea")
            if not effective:
                continue
            keep = ["date"] + sorted(effective)
            df = df[[c for c in keep if c in df.columns]]

        df["fid"] = fid
        chunks.append(df)

    if not chunks:
        raise RuntimeError(f"No QC-passed obs found for source={source}")

    out = pd.concat(chunks, ignore_index=True)

    # Rename obsmet columns → dads names
    rename = {}
    if "ea_compiled" in out.columns:
        rename["ea_compiled"] = "ea"
    if rename:
        out = out.rename(columns=rename)

    # Drop non-metric bookkeeping columns
    drop = [
        c
        for c in (
            "station_key",
            "day_basis",
            "obs_count",
            "coverage_flags",
            "qc_state",
            "qc_reason_codes",
            "qc_rules_version",
            "transform_version",
            "ingest_run_id",
        )
        if c in out.columns
    ]
    if drop:
        out = out.drop(columns=drop)

    out = out.rename(columns={"date": "day"})
    out["day"] = out["day"].dt.normalize()
    out = out.set_index(["fid", "day"]).sort_index()

    n_stations = out.index.get_level_values("fid").nunique()
    print(
        f"  obsmet: loaded {len(out):,} station-days, "
        f"{n_stations:,} stations (source={source})",
        flush=True,
    )
    return out


def load_station_metadata(
    source: str,
    *,
    channel: str = "prod",
    por_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load release manifest metadata for a given source.

    Returns a DataFrame with columns: fid, source, and whatever columns the
    manifest carries (row_count/date_min/date_max in release manifests;
    state/updated_utc/run_id/message in permissive product manifests).

    Parameters
    ----------
    por_path : str | Path | None
        Direct path to the station_por leaf directory.  When provided,
        bypasses the channels/release infrastructure (development mode).
        Production should use build_release().
    """
    if por_path is not None:
        # NOTE: Direct path bypasses channels/release infrastructure.
        manifest_path = Path(por_path) / "manifest.parquet"
    else:
        release_dir = resolve_release(channel)
        manifest_path = release_dir / "manifest.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    mf = pd.read_parquet(manifest_path)
    mf = mf.loc[mf["source"] == source].copy()

    # Extract fid from station_key or key column
    key_col = "station_key" if "station_key" in mf.columns else "key"
    mf["fid"] = mf[key_col].str.split(":", n=1).str[1]
    return mf
