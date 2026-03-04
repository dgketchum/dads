# GridSat-B1 Disk Usage Estimate

## Key Parameters

| Parameter | Value |
|---|---|
| Global grid | 2000 × 5143 (10.3M pixels) |
| PNW subset | 108 × 215 (23.2K pixels) |
| Spatial ratio | 0.23% |
| Raw NC file size | 52.1 MB |
| PNW COG per timestep (3 channels) | 0.125 MB |
| Files per year | ~2920 (8/day × 365) |
| Archive span | 1980–2025 (44 years, 127,563 files) |
| Growing season ratio (Apr–Oct) | 58.6% (214/365 days) |
| Daytime ratio (PNW) | 62.5% (5/8 3-hourly slots) |

## Storage Scenarios

| Scenario | Size |
|---|---|
| Full archive (raw NC, global) | 6.6 TB |
| Full archive (PNW COGs, 3ch) | 15.9 GB |
| 10 yr PNW COGs, 3ch, all hours | 3.6 GB |
| 10 yr PNW COGs, 3ch, growing season only | 2.1 GB |
| 10 yr PNW COGs, 3ch, GS + daytime only | 1.3 GB |
| 5 yr PNW COGs, 3ch, GS + daytime only | 668 MB |

## PNW Daylight Hours by UTC Slot

PNW spans UTC−8 (PST) / UTC−7 (PDT). GridSat provides 3-hourly snapshots at 00, 03, 06, 09, 12, 15, 18, 21 UTC.

| UTC | PDT (Apr–Oct) | PST (Nov–Mar) | Daytime? |
|-----|---------------|----------------|----------|
| 00  | 5:00 PM       | 4:00 PM        | Yes      |
| 03  | 8:00 PM       | 7:00 PM        | Marginal |
| 06  | 11:00 PM      | 10:00 PM       | No       |
| 09  | 2:00 AM       | 1:00 AM        | No       |
| 12  | 5:00 AM       | 4:00 AM        | Marginal |
| 15  | 8:00 AM       | 7:00 AM        | Yes      |
| 18  | 11:00 AM      | 10:00 AM       | Yes      |
| 21  | 2:00 PM       | 1:00 PM        | Yes      |

**Recommended daytime slots (growing season):** 00, 12, 15, 18, 21 UTC (5 of 8).

For IR channels (`irwin_cdr`, `irwvp`), nighttime observations are also valid — cloud-top
temperature is available 24/7. Only `vschn` (visible reflectance) requires daylight.

## Notes

- 1982 and 2017 are missing from the S3 archive
- 2025 is partial (1,944 files as of inventory date)
- COG sizes assume float32; int16 packing would halve storage
- The 3 primary channels are:
  - `vschn` — visible reflectance near 0.6 μm (daytime only)
  - `irwin_cdr` — brightness temperature near 11 μm (CDR quality, 24/7)
  - `irwvp` — brightness temperature near 6.7 μm (water vapor, 24/7)
