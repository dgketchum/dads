"""Unit tests for extract.rs.goes_abi.goes_abi_common."""

from datetime import date, datetime

from extract.rs.goes_abi.goes_abi_common import (
    BAND_MAP,
    BUCKETS,
    CLIP_BOUNDS,
    GOES18_OPERATIONAL,
    HOURS_3H,
    s3_prefix,
    satellites_for_date,
)


class TestS3Prefix:
    """Test S3 key prefix construction."""

    def test_basic_prefix(self):
        dt = datetime(2025, 9, 1, 12)  # Sep 1 = DOY 244
        prefix = s3_prefix("goes16", dt)
        assert prefix == "ABI-L2-MCMIPC/2025/244/12/"

    def test_new_year(self):
        dt = datetime(2025, 1, 1, 0)  # Jan 1 = DOY 1
        prefix = s3_prefix("goes16", dt)
        assert prefix == "ABI-L2-MCMIPC/2025/001/00/"

    def test_leap_year_feb29(self):
        dt = datetime(2024, 2, 29, 6)  # Feb 29 2024 = DOY 60
        prefix = s3_prefix("goes16", dt)
        assert prefix == "ABI-L2-MCMIPC/2024/060/06/"

    def test_dec31_non_leap(self):
        dt = datetime(2025, 12, 31, 21)  # DOY 365
        prefix = s3_prefix("goes18", dt)
        assert prefix == "ABI-L2-MCMIPC/2025/365/21/"

    def test_dec31_leap_year(self):
        dt = datetime(2024, 12, 31, 21)  # DOY 366
        prefix = s3_prefix("goes18", dt)
        assert prefix == "ABI-L2-MCMIPC/2024/366/21/"

    def test_goes18_bucket(self):
        dt = datetime(2023, 6, 15, 9)
        prefix = s3_prefix("goes18", dt)
        assert prefix.startswith("ABI-L2-MCMIPC/2023/")


class TestSatelliteSelection:
    """Test satellite selection logic based on date."""

    def test_pre_goes18(self):
        d = date(2020, 1, 1)
        assert satellites_for_date(d) == ["goes16"]

    def test_goes18_operational_date(self):
        """On the operational date itself, both satellites should be used."""
        assert satellites_for_date(GOES18_OPERATIONAL) == ["goes16", "goes18"]

    def test_post_goes18(self):
        d = date(2023, 6, 15)
        assert satellites_for_date(d) == ["goes16", "goes18"]

    def test_day_before_goes18(self):
        d = GOES18_OPERATIONAL - __import__("datetime").timedelta(days=1)
        assert satellites_for_date(d) == ["goes16"]

    def test_early_goes16(self):
        """GOES-16 era only."""
        d = date(2017, 7, 10)
        assert satellites_for_date(d) == ["goes16"]


class TestBandMapping:
    """Test band mapping completeness."""

    def test_three_channels(self):
        assert len(BAND_MAP) == 3

    def test_expected_channels(self):
        assert set(BAND_MAP.keys()) == {"irwin_cdr", "irwvp", "vschn"}

    def test_abi_variable_names(self):
        assert BAND_MAP["irwin_cdr"] == "CMI_C14"
        assert BAND_MAP["irwvp"] == "CMI_C09"
        assert BAND_MAP["vschn"] == "CMI_C02"


class TestConstants:
    """Test that constants are consistent with GridSat pipeline."""

    def test_hours(self):
        assert HOURS_3H == ["00", "03", "06", "09", "12", "15", "18", "21"]
        assert len(HOURS_3H) == 8

    def test_clip_bounds_keys(self):
        assert set(CLIP_BOUNDS.keys()) == {"lat_min", "lat_max", "lon_min", "lon_max"}

    def test_clip_bounds_values(self):
        assert CLIP_BOUNDS["lat_min"] == 18.2
        assert CLIP_BOUNDS["lat_max"] == 55.4
        assert CLIP_BOUNDS["lon_min"] == -139.4
        assert CLIP_BOUNDS["lon_max"] == -58.0

    def test_buckets(self):
        assert BUCKETS["goes16"] == "noaa-goes16"
        assert BUCKETS["goes18"] == "noaa-goes18"
