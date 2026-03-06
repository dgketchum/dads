try:
    from cube.datasets.zarr_station_dataset import ZarrStationDataset
except ImportError:
    ZarrStationDataset = None

try:
    from cube.datasets.zarr_weather_dataset import ZarrWeatherDataset
except ImportError:
    ZarrWeatherDataset = None

__all__ = ["ZarrStationDataset", "ZarrWeatherDataset"]
