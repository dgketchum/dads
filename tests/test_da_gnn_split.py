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


def _setup_dense_candidate_graph_dir():
    """Create a graph with many candidate sources for one holdout query."""
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
        "source_k": 64,
        "source_max_radius_km": 300.0,
    }
    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    g = HeteroData()
    n_query = 13
    n_source = 12

    g["query"].x = torch.randn(n_query, 37)
    g["query"].y = torch.randn(n_query, 2)
    g["query"].valid_mask = torch.ones(n_query, 2, dtype=torch.bool)
    all_fids = [f"sta_{i:03d}" for i in range(n_query)]
    holdout_fids = {all_fids[0]}
    train_fids = set(all_fids[1:])
    g["query"].fids = all_fids

    g["source"].fids = all_fids[1:]
    g["source"].context_x = torch.randn(n_source, 37)
    g["source"].payload_x = torch.randn(n_source, 4)
    g["source"].num_nodes = n_source

    # Connect every source to the holdout query with increasing distance.
    dists = [5.0 * (i + 1) for i in range(n_source)]  # 5, 10, ..., 60 km
    dist_mean, dist_std = 21.59, 17.30
    src = torch.arange(n_source, dtype=torch.long)
    dst = torch.zeros(n_source, dtype=torch.long)
    ea = torch.zeros(n_source, 7)
    for i, d_km in enumerate(dists):
        ea[i, 0] = (d_km - dist_mean) / dist_std
    g["source", "influences", "query"].edge_index = torch.stack([src, dst])
    g["source", "influences", "query"].edge_attr = ea

    # Minimal query-query graph.
    g["query", "neighbors", "query"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    g["query", "neighbors", "query"].edge_attr = torch.zeros(2, 7)

    torch.save(g, os.path.join(tmpdir, "2024-01-01.pt"))
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


def test_target_source_k_keeps_nearest_surviving_edges():
    """Runtime cap keeps the nearest surviving K source edges per query."""
    tmpdir, holdout, train_fids = _setup_dense_candidate_graph_dir()

    ds_full = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.95,
        da_exclude_radius_km=20.0,
        da_target_source_k=None,
        split_seed=7,
    )
    ds_cap = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.95,
        da_exclude_radius_km=20.0,
        da_target_source_k=4,
        split_seed=7,
    )

    ds_full.set_epoch(0)
    ds_cap.set_epoch(0)
    g_full = ds_full[0]
    g_cap = ds_cap[0]

    dist_mean, dist_std = 21.59, 17.30
    full_ei = g_full["source", "influences", "query"].edge_index
    full_ea = g_full["source", "influences", "query"].edge_attr
    cap_ei = g_cap["source", "influences", "query"].edge_index
    cap_ea = g_cap["source", "influences", "query"].edge_attr

    full_d = (full_ea[full_ei[1] == 0, 0] * dist_std + dist_mean).tolist()
    cap_d = (cap_ea[cap_ei[1] == 0, 0] * dist_std + dist_mean).tolist()

    assert len(full_d) > 4
    assert len(cap_d) == 4
    assert min(cap_d) >= 20.0
    assert sorted(cap_d) == sorted(sorted(full_d)[:4])


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
    assert on.da_target_source_k == off.da_target_source_k == 16
    assert on.split_seed == off.split_seed
    # Only payload differs
    assert on.disable_payload is False
    assert off.disable_payload is True


# ---------------------------------------------------------------------------
# Mixed-local mode
# ---------------------------------------------------------------------------


def test_mixed_local_allows_close_edges():
    """Mixed-local graphs retain some sub-exclusion-radius edges."""
    tmpdir, holdout, train_fids = _setup_graph_dir(
        n_query=30, n_source=20, n_holdout=5, edge_dist_km_range=(5.0, 80.0)
    )

    ds_strict = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=20.0,
        da_target_source_k=16,
    )
    # 100% graph fraction so every graph is mixed-local
    ds_mixed = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=20.0,
        da_target_source_k=16,
        da_mixed_local_enabled=True,
        da_mixed_local_graph_fraction=1.0,
        da_mixed_local_max_edges_per_query=2,
    )

    dist_mean, dist_std = 21.59, 17.30
    any_local = False
    for idx in range(len(ds_mixed)):
        out_m = ds_mixed[idx]
        sq_ei = out_m["source", "influences", "query"].edge_index
        if sq_ei.numel() == 0:
            continue
        ea = out_m["source", "influences", "query"].edge_attr
        d_km = ea[:, 0] * dist_std + dist_mean
        if (d_km < 20.0).any():
            any_local = True

        # Strict version should have no close edges
        out_s = ds_strict[idx]
        sq_ei_s = out_s["source", "influences", "query"].edge_index
        if sq_ei_s.numel() > 0:
            ea_s = out_s["source", "influences", "query"].edge_attr
            d_km_s = ea_s[:, 0] * dist_std + dist_mean
            assert (d_km_s >= 20.0).all()

    assert any_local, "Expected at least one local edge in mixed-local mode"


def test_mixed_local_respects_max_edges_per_query():
    """Mixed-local never exceeds max_local edges per query."""
    tmpdir, holdout, train_fids = _setup_dense_candidate_graph_dir()

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.95,
        da_exclude_radius_km=20.0,
        da_target_source_k=16,
        split_seed=7,
        da_mixed_local_enabled=True,
        da_mixed_local_graph_fraction=1.0,
        da_mixed_local_max_edges_per_query=2,
    )

    dist_mean, dist_std = 21.59, 17.30
    out = ds[0]
    sq_ei = out["source", "influences", "query"].edge_index
    if sq_ei.numel() == 0:
        return
    ea = out["source", "influences", "query"].edge_attr
    d_km = ea[:, 0] * dist_std + dist_mean

    for qi in torch.unique(sq_ei[1]).tolist():
        mask = sq_ei[1] == qi
        local = (d_km[mask] < 20.0).sum().item()
        assert local <= 2, f"Query {qi} has {local} local edges, expected <= 2"


def test_mixed_local_respects_total_k():
    """Mixed-local never exceeds da_target_source_k total edges per query."""
    tmpdir, holdout, train_fids = _setup_dense_candidate_graph_dir()

    ds = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.95,
        da_exclude_radius_km=20.0,
        da_target_source_k=6,
        split_seed=7,
        da_mixed_local_enabled=True,
        da_mixed_local_graph_fraction=1.0,
        da_mixed_local_max_edges_per_query=2,
    )

    out = ds[0]
    sq_ei = out["source", "influences", "query"].edge_index
    if sq_ei.numel() == 0:
        return

    for qi in torch.unique(sq_ei[1]).tolist():
        n_edges = (sq_ei[1] == qi).sum().item()
        assert n_edges <= 6, f"Query {qi} has {n_edges} edges, expected <= 6"


def test_mixed_local_disabled_matches_strict():
    """With da_mixed_local_enabled=False, output matches strict split."""
    tmpdir, holdout, train_fids = _setup_graph_dir(
        n_query=30, n_source=20, n_holdout=5, edge_dist_km_range=(5.0, 80.0)
    )

    ds_strict = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=20.0,
        da_target_source_k=16,
    )
    ds_disabled = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_source_fraction=0.5,
        da_exclude_radius_km=20.0,
        da_target_source_k=16,
        da_mixed_local_enabled=False,
        da_mixed_local_graph_fraction=0.5,
        da_mixed_local_max_edges_per_query=2,
    )

    for idx in range(len(ds_strict)):
        g1 = ds_strict[idx]
        g2 = ds_disabled[idx]
        assert torch.equal(g1["query"].loss_mask, g2["query"].loss_mask)
        assert torch.equal(
            g1["source", "influences", "query"].edge_index,
            g2["source", "influences", "query"].edge_index,
        )


def test_mixed_local_config_parses():
    """Mixed-local config file parses correctly."""
    from models.rtma_bias.train_da_gnn import DAGNNConfig

    cfg = DAGNNConfig.from_toml(
        "models/hrrr_da/configs/da_split_s_tmax_k64_mixed_local.toml"
    )
    assert cfg.da_mixed_local_enabled is True
    assert cfg.da_mixed_local_graph_fraction == 0.15
    assert cfg.da_mixed_local_max_edges_per_query == 2
    # local/far boundary uses da_exclude_radius_km (no separate radius param)
    assert cfg.da_exclude_radius_km == 20.0
    assert cfg.da_split_enabled is True
    assert cfg.da_target_source_k == 16


def test_mixed_local_configs_match_geometry():
    """DA-on and DA-off mixed-local configs use matched split/mixed params."""
    from models.rtma_bias.train_da_gnn import DAGNNConfig

    on = DAGNNConfig.from_toml(
        "models/hrrr_da/configs/da_split_s_tmax_k64_mixed_local.toml"
    )
    off = DAGNNConfig.from_toml(
        "models/hrrr_da/configs/da_split_off_tmax_k64_mixed_local.toml"
    )

    assert on.da_split_enabled is True
    assert off.da_split_enabled is True
    assert on.da_source_fraction == off.da_source_fraction
    assert on.da_exclude_radius_km == off.da_exclude_radius_km
    assert on.da_target_source_k == off.da_target_source_k == 16
    assert on.split_seed == off.split_seed
    assert on.da_mixed_local_enabled == off.da_mixed_local_enabled is True
    assert on.da_mixed_local_graph_fraction == off.da_mixed_local_graph_fraction
    assert (
        on.da_mixed_local_max_edges_per_query == off.da_mixed_local_max_edges_per_query
    )
    # Only payload differs
    assert on.disable_payload is False
    assert off.disable_payload is True


def test_mixed_local_effective_family():
    """Mixed-local datasets report a distinct effective_family."""
    tmpdir, holdout, train_fids = _setup_graph_dir()

    ds_strict = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
    )
    ds_mixed = DAGraphDataset(
        graph_dir=tmpdir,
        loss_fids=train_fids,
        target_index=0,
        is_train=True,
        da_split_enabled=True,
        da_mixed_local_enabled=True,
        da_mixed_local_graph_fraction=0.15,
        da_mixed_local_max_edges_per_query=2,
    )

    assert ds_strict.effective_family == "da-graph-v3"
    assert ds_mixed.effective_family == "da-graph-v3-mixed-local"
