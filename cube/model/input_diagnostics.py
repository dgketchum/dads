import os
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from prep.columns_desc import RS_MISS_FEATURES


def _nan_q(a):
    a = np.asarray(a, dtype=np.float32)
    return tuple(np.nanquantile(a, q=[0.05, 0.50, 0.95]).tolist())


def _load_scaler(path):
    with open(path, "r") as f:
        p = json.load(f)
    bias = np.asarray(p["bias"], dtype=np.float32).reshape(-1)
    scale = np.asarray(p["scale"], dtype=np.float32).reshape(-1)
    names = p.get("feature_names")
    return bias, scale, names


def _list_files(parquet_dir):
    return [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ]


def _flatten_values(dict_like):
    vals = []
    for v in dict_like.values():
        if isinstance(v, list):
            vals.extend(v)
        elif isinstance(v, dict):
            vals.extend(v.values())
    return np.asarray(vals, dtype=np.float32)


def _read_columns(arg):
    fp, cols = arg
    df = pd.read_parquet(fp, columns=cols)
    return {c: df[c].to_numpy(dtype=np.float32, copy=False) for c in cols}


def diagnose_inputs(training_root, variable, num_workers=12):
    target = f"{variable}_obs"
    parquet_dir = os.path.join(training_root, "parquet", target)
    graph_dir = os.path.join(training_root, "graph", target)
    scaler_json = os.path.join(training_root, "scalers", f"{variable}.json")

    files = _list_files(parquet_dir)
    assert files, "no parquet files found"
    # restrict to stations referenced by graph splits (train+val)
    t_attr = os.path.join(graph_dir, "train_edge_attr.json")
    v_attr = os.path.join(graph_dir, "val_edge_attr.json")
    if os.path.exists(t_attr):
        with open(t_attr, "r") as f:
            tr = json.load(f)
    else:
        tr = {}
    if os.path.exists(v_attr):
        with open(v_attr, "r") as f:
            va = json.load(f)
    else:
        va = {}
    keep = set(list(tr.keys()) + list(va.keys()))
    if keep:
        files = [
            fp for fp in files if os.path.splitext(os.path.basename(fp))[0] in keep
        ]
        assert files, "no parquet files match graph splits"
    cols = pd.read_parquet(files[0]).columns.tolist()
    bias, scale, names = _load_scaler(scaler_json)
    if names is not None:
        assert names == cols, "scaler feature_names mismatch"
    idx = {c: i for i, c in enumerate(cols)}

    # Columns used by the model (target + rsun + CDR + GEO) but compute over all cols to be exhaustive.
    # exclude any *_miss columns from diagnostics
    miss_set = {c for c in cols if c.endswith("_miss")}
    miss_set.update({c for c in RS_MISS_FEATURES if c in cols})
    report_cols = [c for c in cols if c not in miss_set]

    # Accumulate data per column (multiprocess)
    data = {c: [] for c in report_cols}
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            args = [(fp, report_cols) for fp in files]
            for out in ex.map(_read_columns, args):
                for c in report_cols:
                    data[c].append(out[c])
    else:
        for fp in files:
            df = pd.read_parquet(fp, columns=report_cols)
            for c in report_cols:
                v = df[c].to_numpy(dtype=np.float32, copy=False)
                data[c].append(v)
    data = {
        c: (np.concatenate(v) if v else np.zeros((0,), dtype=np.float32))
        for c, v in data.items()
    }

    # Print concise stats
    print("\n[Input] columns:", len(report_cols), "files:", len(files))
    print("[Input] target:", target)

    print("\n[Raw vs Scaled] p05/p50/p95 per column:")
    for c in report_cols:
        j = idx[c]
        raw = data[c]
        if raw.size == 0:
            continue
        r05, r50, r95 = _nan_q(raw)
        sc = (raw - bias[j]) / scale[j] + 5e-8
        s05, s50, s95 = _nan_q(sc)
        inv = (sc - 5e-8) * scale[j] + bias[j]
        mad = float(np.nanmax(np.abs(inv - raw))) if inv.size else 0.0
        print(
            f"- {c}: raw=({r05:.3f},{r50:.3f},{r95:.3f}) scaled=({s05:.3f},{s50:.3f},{s95:.3f}) inv_err={mad:.6f}"
        )

    # Distances and bearings ranges (train + val if present)
    d_vals = []
    b_vals = []
    for split in ("train", "val"):
        d_fp = os.path.join(graph_dir, f"{split}_edge_distance.json")
        b_fp = os.path.join(graph_dir, f"{split}_edge_bearing.json")
        if os.path.exists(d_fp):
            with open(d_fp, "r") as f:
                d = json.load(f)
            d_vals.append(_flatten_values(d))
        if os.path.exists(b_fp):
            with open(b_fp, "r") as f:
                b = json.load(f)
            b_vals.append(_flatten_values(b))
    if d_vals:
        d = np.concatenate(d_vals)
        print(
            "\n[Edges] distance_km: min={:.3f} median={:.3f} max={:.3f}".format(
                np.nanmin(d), np.nanmedian(d), np.nanmax(d)
            )
        )
    if b_vals:
        b = np.concatenate(b_vals)
        print(
            "[Edges] bearing_deg: min={:.3f} median={:.3f} max={:.3f}".format(
                np.nanmin(b), np.nanmedian(b), np.nanmax(b)
            )
        )


if __name__ == "__main__":
    training = "/data/ssd2/dads/training"
    var = "tmax"
    diagnose_inputs(training, var, num_workers=8)
# ========================= EOF ====================================================================
