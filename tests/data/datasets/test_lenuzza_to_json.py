import json
from pathlib import Path
from typing import List, Dict, NamedTuple
import importlib.util
import types
import sys

import numpy as np

# Stub pandas required by the script
pd_stub = types.ModuleType("pandas")
pd_stub.DataFrame = object
sys.modules.setdefault("pandas", pd_stub)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Lightweight pff.data.data_empirical.json_schema module
json_schema_spec = importlib.util.spec_from_file_location(
    "json_schema", ROOT / "pff" / "data" / "data_empirical" / "json_schema.py"
)
json_schema = importlib.util.module_from_spec(json_schema_spec)
json_schema_spec.loader.exec_module(json_schema)  # type: ignore[arg-type]
sim_pkg = types.ModuleType("pff")
data_pkg = types.ModuleType("pff.data")
data_emp_pkg = types.ModuleType("pff.data.data_empirical")
data_emp_pkg.json_schema = json_schema
data_pkg.data_empirical = data_emp_pkg
sim_pkg.data = data_pkg
sys.modules.setdefault("pff", sim_pkg)
sys.modules.setdefault("pff.data", data_pkg)
sys.modules.setdefault("pff.data.data_empirical", data_emp_pkg)
sys.modules.setdefault("pff.data.data_empirical.json_schema", json_schema)

SCRIPT_PATH = ROOT / "scripts" / "data" / "lenuzza_to_json.py"
spec = importlib.util.spec_from_file_location("lenuzza_to_json", SCRIPT_PATH)
lenuzza = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lenuzza)  # type: ignore[arg-type]

dataframe_to_study_jsons = lenuzza.dataframe_to_study_jsons
save_study_jsons = lenuzza.save_study_jsons


class _Series(list):
    def astype(self, dtype):
        return _Series([dtype(x) for x in self])

    def to_numpy(self):
        return np.array(self, dtype=float)

    @property
    def iloc(self):
        return self


class _DF:
    """Very small subset of the pandas.DataFrame API used in the script."""

    def __init__(self, rows: List[Dict]):
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def sort_values(self, key: str) -> "_DF":
        return _DF(sorted(self._rows, key=lambda r: r[key]))

    def __getitem__(self, key: str) -> _Series:
        return _Series([r[key] for r in self._rows])

    def groupby(self, keys, sort: bool = False):
        groups = {}
        if isinstance(keys, list):
            for r in self._rows:
                k = tuple(r[k] for k in keys)
                groups.setdefault(k, []).append(r)
        else:
            for r in self._rows:
                k = r[keys]
                groups.setdefault(k, []).append(r)
        for k, rows in groups.items():
            yield k, _DF(rows)


class AICMECompartmentsDataBatch(NamedTuple):
    context_obs: np.ndarray
    context_obs_time: np.ndarray
    context_obs_mask: np.ndarray
    target_obs: np.ndarray
    target_obs_time: np.ndarray
    target_obs_mask: np.ndarray


def _sample_df() -> _DF:
    rows = [
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "A", "time": 0.0, "value": 0.1},
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "A", "time": 1.0, "value": 0.2},
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "A", "time": 2.0, "value": 0.3},
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "B", "time": 0.0, "value": 0.4},
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "B", "time": 1.0, "value": 0.5},
        {"study_name": "S1", "substance_label": "Drug", "subject_name": "B", "time": 2.0, "value": 0.6},
    ]
    return _DF(rows)


def _simple_build(study: Dict) -> AICMECompartmentsDataBatch:
    ctx = study["context"]
    tgt = study["target"]
    c_obs = np.array([c["observations"] for c in ctx], dtype=float)[None, :, :, None]
    c_time = np.array([c["observation_times"] for c in ctx], dtype=float)[None, :, :, None]
    c_mask = np.ones(c_obs.shape[:3], dtype=bool)
    if tgt:
        t_obs = np.array([t["observations"] for t in tgt], dtype=float)[None, :, :, None]
        t_time = np.array([t["observation_times"] for t in tgt], dtype=float)[None, :, :, None]
        t_mask = np.ones(t_obs.shape[:3], dtype=bool)
    else:
        t_obs = np.zeros((1, 0, 0, 1))
        t_time = np.zeros((1, 0, 0, 1))
        t_mask = np.zeros((1, 0, 0), dtype=bool)
    return AICMECompartmentsDataBatch(c_obs, c_time, c_mask, t_obs, t_time, t_mask)


def test_roundtrip_json(tmp_path: Path):
    df = _sample_df()
    study_jsons = dataframe_to_study_jsons(df)
    out = tmp_path / "lenuzza.json"
    save_study_jsons(study_jsons, out)

    with out.open() as f:
        loaded = json.load(f)
    assert loaded == study_jsons


def test_build_batch_from_json(tmp_path: Path):
    df = _sample_df()
    out = tmp_path / "lenuzza.json"
    save_study_jsons(dataframe_to_study_jsons(df), out)

    with out.open() as f:
        study_list = json.load(f)

    batch = _simple_build(study_list[0])
    assert isinstance(batch, AICMECompartmentsDataBatch)
    assert batch.context_obs.shape[0] == 1


def test_all_in_context_default():
    df = _sample_df()
    study_jsons = dataframe_to_study_jsons(df)
    assert len(study_jsons) == 1
    study = study_jsons[0]
    assert study["target"] == []
    assert len(study["context"]) == 2
