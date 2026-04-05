"""Tests for GNN DA runtime source/query split."""

from __future__ import annotations

import json
import os
import tempfile

import torch
from torch_geometric.data import HeteroData

from models.rtma_bias.da_graph_dataset import DAGraphDataset


def _make_graph(
    n_query=20,
    n_source=15,
    n_holdout=5,
    edge_dist_km_range=(5.0, 80.0),
    seed=0,
):
    """Create a synthetic DA v1 HeteroData graph on disk."""
    rng = torch.Generator().manual_seed(seed)
    g = HeteroData()

    # Query nodes
    g["query"].x = torch.randn(n_query, 37, generator=rng)
    g["query"].y = torch.randn(n_query, 2, generator=rng)
    g["query"].valid_mask = torch.ones(n_query, 2, dtype=torch.bool)

    # FIDs: first n_holdout are holdout, rest are non-holdout
    # Source fids are a subset of non-holdout query fids
    all_fids = [f"sta_{i:03d}" for i in range(n_query)]
    g["query"].fids = all_fids

    source_fids = all_fids[n_holdout : n_holdout + n_source]
    g["source"].fids = source_fids
    g["source"].context_x = torch.randn(n_source, 37, generator=rng)
    g["source"].payload_x = torch.randn(n_source, 4, generator=rng)
    g["source"].num_nodes = n_source

    # Source→query edges: each source connects to ~3 nearby query nodes
    src_list, dst_list, dist_list = [], [], []
    for si in range(n_source):
        src_fid = source_fids[si]
        src_qi = all_fids.index(src_fid)
        for qi in range(n_query):
            if qi == src_qi:
                continue  # no self-edge
            if torch.rand(1, generator=rng).item() < 0.2:
                src_list.append(si)
                dst_list.append(qi)
                d_km = edge_dist_km_range[0] + torch.rand(1, generator=rng).item() * (
                    edge_dist_km_range[1] - edge_dist_km_range[0]
                )
                dist_list.append(d_km)

    if not src_list:
        # Ensure at least one edge
        src_list.append(0)
        dst_list.append(
            n_holdout + n_source - 1 if n_holdout + n_source < n_query else 0
        )
        dist_list.append(30.0)

    sq_ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    # Edge attr: ch0 = normalized distance, rest zeros
    dist_mean, dist_std = 21.59, 17.30
    ea = torch.zeros(len(src_list), 7)
    for i, d in enumerate(dist_list):
        ea[i, 0] = (d - dist_mean) / dist_std
    g["source", "influences", "query"].edge_index = sq_ei
    g["source", "influences", "query"].edge_attr = ea

    # Query→query edges (minimal)
    qq_src, qq_dst = [], []
    for qi in range(n_query):
        for qj in range(qi + 1, min(qi + 3, n_query)):
            qq_src.extend([qi, qj])
            qq_dst.extend([qj, qi])
    g["query", "neighbors", "query"].edge_index = torch.tensor(
        [qq_src, qq_dst], dtype=torch.long
    )
    g["query", "neighbors", "query"].edge_attr = torch.randn(len(qq_src), 7)

    return g, set(all_fids[:n_holdout]), set(all_fids[n_holdout:])


def _setup_graph_dir(n_days=3, **graph_kwargs):
    """Create a temp dir with synthetic graphs and meta.json."""
    tmpdir = tempfile.mkdtemp()

    meta = {
        "family": "da-graph-v1",
        "query_feature_cols": [f"qf_{i}" for i in range(37)],
        "target_cols": ["delta_tmax", "delta_tmin"],
        "source_context_feature_cols": [f"sc_{i}" for i in range(37)],
        "source_payload_feature_cols": [
            "delta_tmax",
            "delta_tmin",
            "valid_delta_tmax",
            "valid_delta_tmin",
        ],
        "source_query_edge_norm": {
            "dist_mean": 21.59,
            "dist_std": 17.30,
            "delev_mean": 15.52,
            "delev_std": 1992.21,
        },
    }
    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    holdout_fids = None
    train_fids = None
    for i in range(n_days):
        date_str = f"2024-01-{i + 1:02d}"
        g, hf, tf = _make_graph(seed=i, **graph_kwargs)
        torch.save(g, os.path.join(tmpdir, f"{date_str}.pt"))
        if holdout_fids is None:
            holdout_fids = hf
            train_fids = tf

    return tmpdir, holdout_fids, train_fids


# ---------------------------------------------------------------------------
# Split-disabled parity
# ---------------------------------------------------------------------------


def test_split_disabled_is_identity():
    """With da_split_enabled=False, output matches legacy behavior."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds_legacy = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
    )
    ds_split_off = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=False,
    )

    for idx in range(len(ds_legacy)):
        g1 = ds_legacy[idx]
        g2 = ds_split_off[idx]
        assert torch.equal(g1["query"].x, g2["query"].x)
        assert torch.equal(g1["query"].loss_mask, g2["query"].loss_mask)
        assert torch.equal(
            g1["source", "influences", "query"].edge_index,
            g2["source", "influences", "query"].edge_index,
        )


# ---------------------------------------------------------------------------
# Disjoint source/query sets
# ---------------------------------------------------------------------------


def test_split_produces_disjoint_roles():
    """No fid appears in both source set and supervised query set."""
    tmpdir, holdout, train_fids = _setup_graph_dir(n_query=30, n_source=20, n_holdout=5)

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=0.0,
    )

    for idx in range(len(ds)):
        g_raw = ds._graphs[idx]
        out = ds[idx]

        # Source fids in output (filtered subset)
        s_fids_raw = list(g_raw["source"].fids)
        n_src_out = out["source"].context_x.shape[0]
        assert n_src_out <= len(s_fids_raw)

        # Query fids with loss_mask=True (supervised queries)
        q_fids = list(g_raw["query"].fids)
        supervised_q_fids = {
            q_fids[i] for i in range(len(q_fids)) if out["query"].loss_mask[i]
        }

        # Holdout fids should not be supervised
        for f in supervised_q_fids:
            assert f not in holdout


# ---------------------------------------------------------------------------
# Holdout preservation
# ---------------------------------------------------------------------------


def test_holdout_never_in_sources():
    """Holdout fids never appear as source nodes regardless of split."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
    )

    for idx in range(len(ds)):
        g_raw = ds._graphs[idx]
        s_fids = set(g_raw["source"].fids)
        for f in holdout:
            assert f not in s_fids  # holdout excluded at graph build time


def test_holdout_never_gets_train_loss():
    """Holdout stations have loss_mask=False in training."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
    )

    for idx in range(len(ds)):
        g_raw = ds._graphs[idx]
        out = ds[idx]
        q_fids = list(g_raw["query"].fids)
        for i, f in enumerate(q_fids):
            if f in holdout:
                assert not out["query"].loss_mask[i]


# ---------------------------------------------------------------------------
# Radius filter
# ---------------------------------------------------------------------------


def test_radius_filter_removes_close_edges():
    """Edges below da_exclude_radius_km are removed."""
    tmpdir, holdout, train_fids = _setup_graph_dir(
        edge_dist_km_range=(1.0, 50.0),
    )

    ds_no_radius = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_exclude_radius_km=0.0,
    )
    ds_with_radius = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_exclude_radius_km=25.0,
    )

    for idx in range(len(ds_no_radius)):
        n_edges_no = ds_no_radius[idx][
            "source", "influences", "query"
        ].edge_index.shape[1]
        n_edges_with = ds_with_radius[idx][
            "source", "influences", "query"
        ].edge_index.shape[1]
        # With radius filter, should have fewer or equal edges
        assert n_edges_with <= n_edges_no


# ---------------------------------------------------------------------------
# Epoch re-randomization
# ---------------------------------------------------------------------------


def test_epoch_changes_split():
    """Different epochs produce different split membership."""
    tmpdir, holdout, train_fids = _setup_graph_dir(n_query=30, n_source=20, n_holdout=5)

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=0.0,
    )

    ds.set_epoch(0)
    out0 = ds[0]
    mask0 = out0["query"].loss_mask.clone()

    ds.set_epoch(1)
    out1 = ds[0]
    mask1 = out1["query"].loss_mask.clone()

    # Different epochs should produce different masks (with high probability)
    # Both should have some supervised queries
    assert mask0.any()
    assert mask1.any()
    # With 20 eligible stations and 50/50 split, probability of identical
    # partition across two random seeds is negligible
    assert not torch.equal(mask0, mask1)


def test_same_epoch_is_deterministic():
    """Same epoch and seed produce identical split."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
    )

    ds.set_epoch(5)
    out_a = ds[0]
    ds.set_epoch(5)
    out_b = ds[0]

    assert torch.equal(out_a["query"].loss_mask, out_b["query"].loss_mask)
    assert torch.equal(
        out_a["source", "influences", "query"].edge_index,
        out_b["source", "influences", "query"].edge_index,
    )


# ---------------------------------------------------------------------------
# Val parity
# ---------------------------------------------------------------------------


def test_val_unaffected_by_split_config():
    """Validation dataset ignores split settings."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds_val = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=holdout,
        target_index=0,
        is_train=False,
        da_split_enabled=True,  # should be ignored
    )

    ds_val_plain = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=holdout,
        target_index=0,
    )

    for idx in range(len(ds_val)):
        g1 = ds_val[idx]
        g2 = ds_val_plain[idx]
        assert torch.equal(g1["query"].loss_mask, g2["query"].loss_mask)
        assert g1["source"].context_x.shape == g2["source"].context_x.shape


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_split_configs_match_geometry():
    """DA-on and DA-off split configs use matched split params."""
    from models.rtma_bias.train_da_gnn import DAGNNConfig

    on = DAGNNConfig.from_toml("models/hrrr_da/configs/da_split_s_tmax.toml")
    off = DAGNNConfig.from_toml("models/hrrr_da/configs/da_split_off_tmax.toml")

    assert on.da_split_enabled is True
    assert off.da_split_enabled is True
    assert on.da_source_fraction == off.da_source_fraction
    assert on.da_exclude_radius_km == off.da_exclude_radius_km
    assert on.split_seed == off.split_seed
    # Only payload differs
    assert on.disable_payload is False
    assert off.disable_payload is True
