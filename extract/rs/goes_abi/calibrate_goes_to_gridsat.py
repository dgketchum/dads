"""Cross-calibrate GOES ABI brightness temperatures to the GridSat-B1 baseline.

Builds a per-channel quantile-mapping transfer function from matched pixel
pairs during the overlap period (2017-07 to 2025-08).  The resulting mapping
tables are serialized to a JSON artifact that the downloader applies before
writing COGs.

Usage
-----
  uv run python -m extract.rs.goes_abi.calibrate_goes_to_gridsat \
      --goes-dir /nas/dads/rs/goes_abi \
      --gridsat-dir /nas/dads/rs/gridsat_b1 \
      --out artifacts/goes_to_gridsat_qmap.json \
      --sample-frac 0.05 --n-quantiles 201
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, timedelta

import numpy as np
import rasterio

from extract.rs.goes_abi.goes_abi_common import BAND_MAP, HOURS_3H

# ── overlap period ────────────────────────────────────────────────────────────

OVERLAP_START = date(2017, 7, 10)
OVERLAP_END = date(2025, 8, 31)

_CHANNELS = list(BAND_MAP.keys())


# ── sampling ──────────────────────────────────────────────────────────────────


def _collect_matched_pairs(
    goes_dir: str,
    gridsat_dir: str,
    sample_frac: float,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect matched pixel pairs from GOES and GridSat COGs.

    Returns {channel: (goes_values, gridsat_values)} with NaN-free arrays.
    Stratifies sampling across months and hours.
    """
    rng = random.Random(seed)
    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {ch: [] for ch in _CHANNELS}

    # Build date list
    d = OVERLAP_START
    dates: list[date] = []
    while d <= OVERLAP_END:
        dates.append(d)
        d += timedelta(days=1)

    for d in dates:
        ds_label = d.strftime("%Y%m%d")
        year = d.strftime("%Y")

        for hh in HOURS_3H:
            for ch in _CHANNELS:
                goes_path = os.path.join(
                    goes_dir, year, f"goes_abi_{ds_label}_{hh}00_{ch}.tif"
                )
                gridsat_path = os.path.join(
                    gridsat_dir, year, f"gridsat_b1_{ds_label}_{hh}00_{ch}.tif"
                )

                if not os.path.exists(goes_path) or not os.path.exists(gridsat_path):
                    continue

                with (
                    rasterio.open(goes_path) as g_src,
                    rasterio.open(gridsat_path) as gs_src,
                ):
                    g_data = g_src.read(1).ravel()
                    gs_data = gs_src.read(1).ravel()

                # Mask NaNs in either source
                valid = ~(np.isnan(g_data) | np.isnan(gs_data))
                g_valid = g_data[valid]
                gs_valid = gs_data[valid]

                if len(g_valid) == 0:
                    continue

                # Subsample
                n_keep = max(1, int(len(g_valid) * sample_frac))
                idx = sorted(rng.sample(range(len(g_valid)), min(n_keep, len(g_valid))))
                pairs[ch].append((g_valid[idx], gs_valid[idx]))

        if d.day == 1:
            total = sum(sum(len(a) for a, _ in v) for v in pairs.values())
            print(f"  {d}: {total:,} pairs collected so far", flush=True)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch in _CHANNELS:
        if not pairs[ch]:
            continue
        goes_all = np.concatenate([a for a, _ in pairs[ch]])
        gridsat_all = np.concatenate([b for _, b in pairs[ch]])
        result[ch] = (goes_all, gridsat_all)

    return result


# ── quantile mapping ──────────────────────────────────────────────────────────


def _build_qmap(
    goes_vals: np.ndarray,
    gridsat_vals: np.ndarray,
    n_quantiles: int = 201,
) -> dict:
    """Build quantile-mapping breakpoints from matched value arrays."""
    quantile_levels = np.linspace(0.0, 1.0, n_quantiles)
    goes_bp = np.quantile(goes_vals, quantile_levels).tolist()
    gridsat_bp = np.quantile(gridsat_vals, quantile_levels).tolist()
    return {
        "goes_breakpoints": goes_bp,
        "gridsat_breakpoints": gridsat_bp,
        "n_pairs": int(len(goes_vals)),
        "n_quantiles": n_quantiles,
    }


def _diagnostics(
    goes_vals: np.ndarray,
    gridsat_vals: np.ndarray,
    channel: str,
    qmap_entry: dict,
) -> None:
    """Print pre- and post-calibration diagnostics for a channel."""
    diff_raw = gridsat_vals - goes_vals
    bias_raw = np.mean(diff_raw)
    rmse_raw = np.sqrt(np.mean(diff_raw**2))
    max_dev_raw = np.max(np.abs(diff_raw))

    goes_bp = np.array(qmap_entry["goes_breakpoints"])
    gridsat_bp = np.array(qmap_entry["gridsat_breakpoints"])
    mapped = np.interp(goes_vals, goes_bp, gridsat_bp)

    diff_cal = gridsat_vals - mapped
    bias_cal = np.mean(diff_cal)
    rmse_cal = np.sqrt(np.mean(diff_cal**2))
    max_dev_cal = np.max(np.abs(diff_cal))

    print(f"\n  {channel}  ({len(goes_vals):,} pairs)")
    print(
        f"    Raw:        bias={bias_raw:+.4f}  RMSE={rmse_raw:.4f}  max={max_dev_raw:.4f}"
    )
    print(
        f"    Calibrated: bias={bias_cal:+.4f}  RMSE={rmse_cal:.4f}  max={max_dev_cal:.4f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GOES→GridSat quantile-mapping calibration artifact"
    )
    parser.add_argument(
        "--goes-dir", default="/nas/dads/rs/goes_abi", help="Root dir for GOES ABI COGs"
    )
    parser.add_argument(
        "--gridsat-dir",
        default="/nas/dads/rs/gridsat_b1",
        help="Root dir for GridSat COGs",
    )
    parser.add_argument(
        "--out",
        default="artifacts/goes_to_gridsat_qmap.json",
        help="Output JSON artifact path",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        help="Fraction of valid pixels to sample per slot (default 0.05)",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=201,
        help="Number of quantile levels (default 201)",
    )
    args = parser.parse_args()

    print(f"Collecting matched pairs (sample_frac={args.sample_frac}) ...", flush=True)
    pairs = _collect_matched_pairs(args.goes_dir, args.gridsat_dir, args.sample_frac)

    if not pairs:
        print(
            "ERROR: no matched pairs found. Ensure both GOES and GridSat COGs exist "
            "for the overlap period (2017-07 to 2025-08)."
        )
        return

    artifact: dict = {}
    for ch in _CHANNELS:
        if ch not in pairs:
            print(f"  WARNING: no pairs for {ch}, skipping")
            continue

        goes_vals, gridsat_vals = pairs[ch]
        qmap_entry = _build_qmap(goes_vals, gridsat_vals, args.n_quantiles)
        artifact[ch] = qmap_entry
        _diagnostics(goes_vals, gridsat_vals, ch, qmap_entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nCalibration artifact written to {args.out}")


if __name__ == "__main__":
    main()
