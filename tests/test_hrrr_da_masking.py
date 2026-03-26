from __future__ import annotations

import json

import pandas as pd
import torch
from torch_geometric.data import Data

from models.components.edge_gated_gnn import EdgeGatedGNN
from models.hrrr_da.hrrr_dataset import HRRRGraphDataset, PrecomputedHRRRDataset


def test_edge_gated_gnn_masks_self_innovations_without_neighbors():
    torch.manual_seed(0)
    model = EdgeGatedGNN(
        node_dim=4,
        edge_dim=7,
        hidden_dim=8,
        n_hops=1,
        use_graph=True,
        out_dim=1,
        dropout=0.0,
    )

    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 7), dtype=torch.float32)

    x1 = torch.tensor([[0.2, -0.4, 8.0, -3.0]], dtype=torch.float32)
    x2 = torch.tensor([[0.2, -0.4, -5.0, 6.0]], dtype=torch.float32)

    out_raw_1 = model(x1, edge_index=edge_index, edge_attr=edge_attr)
    out_raw_2 = model(x2, edge_index=edge_index, edge_attr=edge_attr)
    assert not torch.allclose(out_raw_1, out_raw_2)

    out_masked_1 = model(
        x1,
        edge_index=edge_index,
        edge_attr=edge_attr,
        innovation_indices=[2, 3],
        innovation_fill_values=[-1.5, 0.75],
    )
    out_masked_2 = model(
        x2,
        edge_index=edge_index,
        edge_attr=edge_attr,
        innovation_indices=[2, 3],
        innovation_fill_values=[-1.5, 0.75],
    )
    assert torch.allclose(out_masked_1, out_masked_2)


def test_hrrr_graph_dataset_uses_transductive_holdout_loss_masks(tmp_path):
    day = pd.Timestamp("2024-01-01")
    table_path = tmp_path / "station_day.parquet"
    stations_csv = tmp_path / "stations.csv"

    df = pd.DataFrame(
        [
            {
                "fid": "A",
                "day": day,
                "latitude": 45.0,
                "longitude": -120.0,
                "elevation": 500.0,
                "slope": 0.1,
                "aspect_sin": 0.0,
                "aspect_cos": 1.0,
                "tpi_4": 0.0,
                "tpi_10": 0.0,
                "ugrd_hrrr": 1.0,
                "vgrd_hrrr": 0.5,
                "wind_hrrr": 1.2,
                "tmp_hrrr": 10.0,
                "dpt_hrrr": 6.0,
                "pres_hrrr": 900.0,
                "tcdc_hrrr": 0.0,
                "ea_hrrr": 1.0,
                "dswrf_hrrr": 150.0,
                "hpbl_hrrr": 500.0,
                "spfh_hrrr": 0.005,
                "tmp_dpt_diff": 4.0,
                "doy_sin": 0.1,
                "doy_cos": 0.9,
                "delta_tmax": 2.0,
            },
            {
                "fid": "B",
                "day": day,
                "latitude": 45.1,
                "longitude": -120.1,
                "elevation": 600.0,
                "slope": 0.2,
                "aspect_sin": 0.1,
                "aspect_cos": 0.9,
                "tpi_4": 0.1,
                "tpi_10": 0.1,
                "ugrd_hrrr": 0.8,
                "vgrd_hrrr": 0.3,
                "wind_hrrr": 0.9,
                "tmp_hrrr": 9.0,
                "dpt_hrrr": 5.0,
                "pres_hrrr": 890.0,
                "tcdc_hrrr": 10.0,
                "ea_hrrr": 0.9,
                "dswrf_hrrr": 140.0,
                "hpbl_hrrr": 450.0,
                "spfh_hrrr": 0.004,
                "tmp_dpt_diff": 4.0,
                "doy_sin": 0.1,
                "doy_cos": 0.9,
                "delta_tmax": -1.0,
            },
        ]
    )
    df.to_parquet(table_path, index=False)
    pd.DataFrame(
        [
            {"fid": "A", "latitude": 45.0, "longitude": -120.0, "elevation": 500.0},
            {"fid": "B", "latitude": 45.1, "longitude": -120.1, "elevation": 600.0},
        ]
    ).to_csv(stations_csv, index=False)

    train_ds = HRRRGraphDataset(
        table_path=str(table_path),
        stations_csv=str(stations_csv),
        use_sx=False,
        use_flow_terrain=False,
        use_innovations=False,
        train_days={day},
        exclude_fids={"B"},
        is_val=False,
    )
    val_ds = HRRRGraphDataset(
        table_path=str(table_path),
        stations_csv=str(stations_csv),
        use_sx=False,
        use_flow_terrain=False,
        use_innovations=False,
        train_days={day},
        exclude_fids={"B"},
        is_val=True,
        norm_stats=train_ds.norm_stats,
    )

    train_graph = train_ds[0]
    val_graph = val_ds[0]

    # Non-innovation mode: inductive holdout removes excluded stations
    assert train_graph.fids == ["A"]
    assert val_graph.fids == ["A"]
    assert train_graph.loss_mask.tolist() == [True]
    assert val_graph.loss_mask.tolist() == [True]


def test_precomputed_hrrr_dataset_uses_transductive_holdout_loss_masks(tmp_path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()

    with open(graph_dir / "meta.json", "w") as f:
        json.dump({"all_feature_cols": ["feat"], "target_cols": ["delta_tmax"]}, f)

    graph = Data(
        x=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        y=torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 7), dtype=torch.float32),
    )
    graph.fids = ["A", "B"]
    torch.save(graph, graph_dir / "2024-01-01.pt")

    train_ds = PrecomputedHRRRDataset(
        graph_dir=str(graph_dir),
        train_days={pd.Timestamp("2024-01-01")},
        holdout_fids={"B"},
        is_val=False,
    )
    val_ds = PrecomputedHRRRDataset(
        graph_dir=str(graph_dir),
        train_days={pd.Timestamp("2024-01-01")},
        holdout_fids={"B"},
        is_val=True,
    )

    assert train_ds[0].loss_mask.tolist() == [True, False]
    assert val_ds[0].loss_mask.tolist() == [False, True]
