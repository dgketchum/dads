"""Unit tests for extract.rs.geosat_reader — path resolution logic.

Uses pytest tmp_path to create mock directory structures with empty files.
No NAS access required.
"""

from __future__ import annotations

import os
from datetime import date


from extract.rs.geosat_reader import (
    CHANNELS,
    GOES_START,
    GRIDSAT_END,
    resolve_cog_path,
    resolve_day_paths,
    source_for_date,
)
from extract.rs.goes_abi.goes_abi_common import HOURS_3H


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


class TestResolveCogPath:
    def test_gridsat_before_cutoff(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2020, 6, 15)
        _touch(os.path.join(gs_root, "2020", "gridsat_b1_20200615_1200_irwin_cdr.tif"))
        result = resolve_cog_path(d, "12", "irwin_cdr", gs_root, goes_root)
        assert result is not None
        assert "gridsat_b1" in result

    def test_goes_after_cutoff(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2025, 10, 1)
        _touch(os.path.join(goes_root, "2025", "goes_abi_20251001_0900_irwin_cdr.tif"))
        result = resolve_cog_path(d, "09", "irwin_cdr", gs_root, goes_root)
        assert result is not None
        assert "goes_abi" in result

    def test_overlap_prefers_gridsat(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2020, 1, 1)
        _touch(os.path.join(gs_root, "2020", "gridsat_b1_20200101_0000_irwin_cdr.tif"))
        _touch(os.path.join(goes_root, "2020", "goes_abi_20200101_0000_irwin_cdr.tif"))
        result = resolve_cog_path(d, "00", "irwin_cdr", gs_root, goes_root)
        assert "gridsat_b1" in result

    def test_overlap_falls_through_to_goes(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2020, 1, 1)
        # Only GOES exists
        _touch(os.path.join(goes_root, "2020", "goes_abi_20200101_0000_irwin_cdr.tif"))
        result = resolve_cog_path(d, "00", "irwin_cdr", gs_root, goes_root)
        assert result is not None
        assert "goes_abi" in result

    def test_both_missing_returns_none(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        result = resolve_cog_path(
            date(2020, 1, 1), "00", "irwin_cdr", gs_root, goes_root
        )
        assert result is None

    def test_boundary_date_gridsat_end(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        # GRIDSAT_END = 2025-08-31 should still prefer gridsat
        d = GRIDSAT_END
        ds = d.strftime("%Y%m%d")
        _touch(
            os.path.join(gs_root, str(d.year), f"gridsat_b1_{ds}_0000_irwin_cdr.tif")
        )
        result = resolve_cog_path(d, "00", "irwin_cdr", gs_root, goes_root)
        assert "gridsat_b1" in result

    def test_day_after_gridsat_end(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2025, 9, 1)
        _touch(os.path.join(goes_root, "2025", "goes_abi_20250901_0000_irwin_cdr.tif"))
        result = resolve_cog_path(d, "00", "irwin_cdr", gs_root, goes_root)
        assert "goes_abi" in result

    def test_path_format_gridsat(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2019, 3, 5)
        expected = os.path.join(gs_root, "2019", "gridsat_b1_20190305_0600_vschn.tif")
        _touch(expected)
        result = resolve_cog_path(d, "06", "vschn", gs_root, goes_root)
        assert result == expected

    def test_path_format_goes(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2025, 12, 25)
        expected = os.path.join(goes_root, "2025", "goes_abi_20251225_1500_irwvp.tif")
        _touch(expected)
        result = resolve_cog_path(d, "15", "irwvp", gs_root, goes_root)
        assert result == expected


class TestResolveDayPaths:
    def test_all_slots_present(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2020, 7, 4)
        ds = d.strftime("%Y%m%d")
        for hh in HOURS_3H:
            for ch in CHANNELS:
                _touch(
                    os.path.join(gs_root, "2020", f"gridsat_b1_{ds}_{hh}00_{ch}.tif")
                )
        paths = resolve_day_paths(d, gs_root, goes_root)
        assert len(paths) == 24  # 8 hours x 3 channels

    def test_partial_day(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        d = date(2020, 7, 4)
        ds = d.strftime("%Y%m%d")
        # Only 2 hours, all channels
        for hh in ["06", "12"]:
            for ch in CHANNELS:
                _touch(
                    os.path.join(gs_root, "2020", f"gridsat_b1_{ds}_{hh}00_{ch}.tif")
                )
        paths = resolve_day_paths(d, gs_root, goes_root)
        assert len(paths) == 6  # 2 hours x 3 channels

    def test_empty_day(self, tmp_path):
        gs_root = str(tmp_path / "gridsat")
        goes_root = str(tmp_path / "goes")
        paths = resolve_day_paths(date(2020, 7, 4), gs_root, goes_root)
        assert len(paths) == 0


class TestSourceForDate:
    def test_pre_overlap(self):
        assert source_for_date(date(2010, 1, 1)) == "gridsat"

    def test_overlap(self):
        assert source_for_date(date(2020, 1, 1)) == "overlap"

    def test_post_gridsat(self):
        assert source_for_date(date(2025, 9, 1)) == "goes"

    def test_goes_start_boundary(self):
        assert source_for_date(GOES_START) == "overlap"

    def test_gridsat_end_boundary(self):
        assert source_for_date(GRIDSAT_END) == "overlap"

    def test_day_before_goes_start(self):
        d = date(GOES_START.year, GOES_START.month, GOES_START.day - 1)
        assert source_for_date(d) == "gridsat"
