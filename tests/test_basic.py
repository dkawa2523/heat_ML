from pathlib import Path

import torch

from celltemp.config import load_config, project_root_from_config
from celltemp.data import load_cases, parse_condition
from celltemp.features import WindowDataset
from celltemp.models import build_model
from celltemp.preprocess import fit_preprocessor
from celltemp.thermal_graph import load_thermal_graph


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "config_train.yaml"


def test_parse_condition():
    cfg = load_config(CFG_PATH)
    cond = parse_condition("temp_10_120_70.csv", cfg["data"]["filename_regex"])
    assert cond == {"brine": 10.0, "heater": 120.0, "plasma": 70.0}


def test_graph_loads_from_tables():
    cfg = load_config(CFG_PATH)
    graph = load_thermal_graph(cfg["model"]["graph"], ROOT, cfg["data"]["sensor_cols"], cfg["data"]["control_cols"])
    assert graph.edge_mask.shape == (4, 4)
    assert graph.source_weight.shape == (4, 3)
    assert graph.edge_mask.sum() > 0


def test_all_models_forward_shape():
    cfg = load_config(CFG_PATH)
    root = project_root_from_config(CFG_PATH)
    cases = load_cases(cfg["data"], root)
    pre = fit_preprocessor(cases[:3], cfg["data"]["sensor_cols"], cfg["data"]["control_cols"], cfg.get("features", {}))
    ds = WindowDataset(cases[:3], cfg["data"]["sensor_cols"], cfg["data"]["control_cols"], pre, history=3, feature_cfg=cfg.get("features", {}))
    batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    for name in ["linear_rc", "mlp", "cnn1d", "gru", "lstm", "tcn", "thermal_state_space", "graph_rc"]:
        model_cfg = dict(cfg["model"])
        model_cfg["name"] = name
        model = build_model(model_cfg, 4, 3, 3, cfg["data"]["sensor_cols"], cfg["data"]["control_cols"], root)
        out = model(batch)
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (1, 4)
