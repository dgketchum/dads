"""Tests for Stage B patch-assim trainer defaults."""

import torch

from models.hrrr_da.patch_assim_dataset import collate_day_tile_batch, collate_patch
from models.hrrr_da.train_patch_assim import (
    DayTileResamplingCallback,
    PatchAssimConfig,
    _resolve_center_limits,
    _resolve_loader_settings,
    _resolve_train_dataset_kwargs,
)


def test_benchmark_mode_defaults_val_loader_to_single_worker():
    cfg = PatchAssimConfig(num_workers=4, benchmark_mode=True)

    train_nw, val_nw, train_persistent, val_persistent = _resolve_loader_settings(cfg)

    assert train_nw == 4
    assert val_nw == 1
    assert train_persistent is True
    assert val_persistent is False


def test_explicit_val_worker_override_is_preserved():
    cfg = PatchAssimConfig(
        num_workers=4,
        benchmark_mode=True,
        train_num_workers=3,
        val_num_workers=0,
        train_persistent_workers=False,
        val_persistent_workers=True,
    )

    train_nw, val_nw, train_persistent, val_persistent = _resolve_loader_settings(cfg)

    assert train_nw == 3
    assert val_nw == 0
    assert train_persistent is False
    assert val_persistent is False


def test_center_limits_resolve_train_and_val_overrides():
    cfg = PatchAssimConfig(
        centers_per_day=64,
        train_centers_per_day=32,
        val_centers_per_day=8,
    )

    train_centers, val_centers = _resolve_center_limits(cfg)

    assert train_centers == 32
    assert val_centers == 8


def test_tile_train_mode_resolves_tile_dataset_kwargs():
    cfg = PatchAssimConfig(
        train_sample_mode="tile",
        train_tile_stride=32,
        train_tiles_per_day=24,
        seed=7,
    )

    dataset_cls, kwargs = _resolve_train_dataset_kwargs(cfg)

    assert dataset_cls.__name__ == "HRRRTilePatchDataset"
    assert kwargs == {
        "tile_stride": 32,
        "tiles_per_day": 24,
        "tile_sampling_seed": 7,
    }


def test_day_tile_mode_resolves_day_tile_dataset_kwargs():
    cfg = PatchAssimConfig(
        train_sample_mode="day_tile",
        train_tile_stride=64,
        train_tiles_per_day=16,
        seed=99,
    )

    dataset_cls, kwargs = _resolve_train_dataset_kwargs(cfg)

    assert dataset_cls.__name__ == "HRRRDayTileBatchDataset"
    assert kwargs == {
        "tile_stride": 64,
        "tiles_per_day": 16,
        "tile_sampling_seed": 99,
    }


def _make_sample(n_sta=3, n_targets=1, C=4, H=64, W=64):
    """Create a synthetic Stage B 7-tuple sample."""
    return (
        torch.randn(C, H, W),
        torch.randint(0, H, (n_sta,)),
        torch.randint(0, W, (n_sta,)),
        torch.randn(n_sta, n_targets),
        torch.ones(n_sta, n_targets, dtype=torch.bool),
        torch.zeros(n_sta, dtype=torch.bool),
        torch.zeros(n_sta, dtype=torch.bool),
    )


def test_collate_day_tile_batch_flattens_to_standard_format():
    """Day-centric collate should flatten K samples from N days into one batch."""
    day1 = [_make_sample(n_sta=2), _make_sample(n_sta=4)]
    day2 = [_make_sample(n_sta=3)]
    batch = [day1, day2]  # 2 days, 3 total tiles

    result = collate_day_tile_batch(batch)
    x, sta_rows, sta_cols, sta_targets, sta_valid, sta_holdout, sta_is_center = result

    assert x.shape[0] == 3  # 3 tiles total
    assert sta_rows.shape[0] == 3
    max_sta = max(2, 4, 3)
    assert sta_rows.shape[1] == max_sta


def test_collate_day_tile_batch_matches_collate_patch():
    """Flattened day-tile collation should produce identical output to collate_patch."""
    samples = [_make_sample(n_sta=2), _make_sample(n_sta=5), _make_sample(n_sta=1)]

    # Flat collation
    flat_result = collate_patch(samples)
    # Day-centric collation (all in one day payload)
    day_result = collate_day_tile_batch([samples])

    for a, b in zip(flat_result, day_result):
        assert torch.equal(a, b)


def test_day_tile_resampling_callback_calls_set_epoch():
    """DayTileResamplingCallback should forward epoch to both sampler and dataset."""

    class FakeSampler:
        def __init__(self):
            self.epoch = -1

        def set_epoch(self, e):
            self.epoch = e

    class FakeDataset:
        def __init__(self):
            self.epoch = -1

        def set_epoch(self, e):
            self.epoch = e

    class FakeTrainer:
        current_epoch = 7

    sampler = FakeSampler()
    dataset = FakeDataset()
    cb = DayTileResamplingCallback(sampler, dataset)
    cb.on_train_epoch_start(FakeTrainer(), None)

    assert sampler.epoch == 7
    assert dataset.epoch == 7


def test_shared_epoch_visible_across_processes():
    """mp.Value epoch must be readable from a child process after main-process set_epoch."""
    import multiprocessing as mp_mod

    shared = mp_mod.Value("i", 0)

    def _read_epoch(shared_val, result_queue):
        result_queue.put(shared_val.value)

    q = mp_mod.Queue()
    # Initial value visible in child
    p = mp_mod.Process(target=_read_epoch, args=(shared, q))
    p.start()
    p.join()
    assert q.get() == 0

    # After update, child sees new value
    shared.value = 5
    p2 = mp_mod.Process(target=_read_epoch, args=(shared, q))
    p2.start()
    p2.join()
    assert q.get() == 5
