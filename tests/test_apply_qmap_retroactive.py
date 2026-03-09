"""Unit tests for extract.rs.goes_abi.apply_qmap_retroactive."""

import os
from datetime import date

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from extract.rs.goes_abi.apply_qmap_retroactive import (
    _OVERLAP_END,
    _apply_qmap_to_file,
    _collect_files,
    _parse_filename,
)


class TestParseFilename:
    """Test GOES ABI COG filename parsing."""

    def test_irwin_cdr(self):
        result = _parse_filename("goes_abi_20251001_0600_irwin_cdr.tif")
        assert result == (date(2025, 10, 1), "irwin_cdr")

    def test_irwvp(self):
        result = _parse_filename("goes_abi_20250901_1200_irwvp.tif")
        assert result == (date(2025, 9, 1), "irwvp")

    def test_vschn(self):
        result = _parse_filename("goes_abi_20260101_0000_vschn.tif")
        assert result == (date(2026, 1, 1), "vschn")

    def test_full_path(self):
        result = _parse_filename(
            "/nas/dads/rs/goes_abi/2025/goes_abi_20251225_2100_irwvp.tif"
        )
        assert result == (date(2025, 12, 25), "irwvp")

    def test_non_matching(self):
        assert _parse_filename("gridsat_b1_20250901_0000_irwin_cdr.tif") is None

    def test_no_extension(self):
        assert _parse_filename("goes_abi_20250901_0000_irwvp") is None

    def test_wrong_extension(self):
        assert _parse_filename("goes_abi_20250901_0000_irwvp.nc") is None


class TestCollectFiles:
    """Test file collection with overlap-period filtering."""

    def _make_cog(self, path: str) -> None:
        """Write a tiny 1-band GeoTIFF."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": from_bounds(-100, 40, -99, 41, 2, 2),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.array([[280.0, 285.0], [290.0, 295.0]], dtype=np.float32), 1)

    def test_skips_overlap_period(self, tmp_path):
        overlap_file = tmp_path / "goes_abi_20250801_0600_irwin_cdr.tif"
        post_file = tmp_path / "goes_abi_20250901_0600_irwin_cdr.tif"
        self._make_cog(str(overlap_file))
        self._make_cog(str(post_file))

        result = _collect_files(str(tmp_path))
        assert len(result) == 1
        assert result[0][1] == "irwin_cdr"
        assert "20250901" in result[0][0]

    def test_skips_overlap_boundary(self, tmp_path):
        """Files on _OVERLAP_END itself should be skipped."""
        boundary_file = tmp_path / "goes_abi_20250831_1200_irwvp.tif"
        self._make_cog(str(boundary_file))

        result = _collect_files(str(tmp_path))
        assert len(result) == 0

    def test_includes_day_after_overlap(self, tmp_path):
        post_file = tmp_path / "goes_abi_20250901_0000_vschn.tif"
        self._make_cog(str(post_file))

        result = _collect_files(str(tmp_path))
        assert len(result) == 1
        assert result[0][1] == "vschn"

    def test_skips_unknown_channel(self, tmp_path):
        bad_file = tmp_path / "goes_abi_20251001_0600_fake_channel.tif"
        self._make_cog(str(bad_file))

        result = _collect_files(str(tmp_path))
        assert len(result) == 0

    def test_skips_non_tif(self, tmp_path):
        parquet_file = tmp_path / "manifest.parquet"
        parquet_file.write_text("not a tif")

        result = _collect_files(str(tmp_path))
        assert len(result) == 0

    def test_walks_subdirs(self, tmp_path):
        subdir = tmp_path / "2025"
        self._make_cog(str(subdir / "goes_abi_20251015_0900_irwvp.tif"))
        self._make_cog(str(subdir / "goes_abi_20251015_0900_irwin_cdr.tif"))

        result = _collect_files(str(tmp_path))
        assert len(result) == 2
        channels = {ch for _, ch in result}
        assert channels == {"irwvp", "irwin_cdr"}

    def test_empty_dir(self, tmp_path):
        assert _collect_files(str(tmp_path)) == []


class TestApplyQmapToFile:
    """Test in-place qmap application to a COG file."""

    def _write_cog(self, path: str, data: np.ndarray) -> None:
        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": from_bounds(-100, 40, -99, 41, data.shape[1], data.shape[0]),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    def test_applies_linear_mapping(self, tmp_path):
        """A simple linear qmap should shift values."""
        path = str(tmp_path / "test.tif")
        data = np.array([[200.0, 250.0], [300.0, 350.0]], dtype=np.float32)
        self._write_cog(path, data)

        # Identity-ish qmap with a +5 offset
        qmap = {
            "irwin_cdr": {
                "goes_breakpoints": [100.0, 200.0, 300.0, 400.0],
                "gridsat_breakpoints": [105.0, 205.0, 305.0, 405.0],
            }
        }

        status = _apply_qmap_to_file(path, "irwin_cdr", qmap)
        assert status == "calibrated"

        with rasterio.open(path) as src:
            result = src.read(1)

        np.testing.assert_allclose(result, data + 5.0, atol=0.01)

    def test_preserves_nan(self, tmp_path):
        path = str(tmp_path / "test.tif")
        data = np.array([[np.nan, 250.0], [300.0, np.nan]], dtype=np.float32)
        self._write_cog(path, data)

        qmap = {
            "irwin_cdr": {
                "goes_breakpoints": [100.0, 400.0],
                "gridsat_breakpoints": [100.0, 400.0],
            }
        }

        status = _apply_qmap_to_file(path, "irwin_cdr", qmap)
        assert status == "calibrated"

        with rasterio.open(path) as src:
            result = src.read(1)

        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])
        assert not np.isnan(result[0, 1])
        assert not np.isnan(result[1, 0])

    def test_all_nan_returns_status(self, tmp_path):
        path = str(tmp_path / "test.tif")
        data = np.full((2, 2), np.nan, dtype=np.float32)
        self._write_cog(path, data)

        qmap = {
            "irwin_cdr": {
                "goes_breakpoints": [100.0, 400.0],
                "gridsat_breakpoints": [100.0, 400.0],
            }
        }

        status = _apply_qmap_to_file(path, "irwin_cdr", qmap)
        assert status == "all_nan"

    def test_missing_channel_in_qmap(self, tmp_path):
        path = str(tmp_path / "test.tif")
        self._write_cog(path, np.ones((2, 2), dtype=np.float32) * 280.0)

        status = _apply_qmap_to_file(path, "irwin_cdr", {})
        assert status == "no_qmap_entry"


class TestOverlapEnd:
    """Verify the overlap-end constant is correct."""

    def test_overlap_end_date(self):
        assert _OVERLAP_END == date(2025, 8, 31)
