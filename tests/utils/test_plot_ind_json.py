import json
import pathlib
import pytest
import sys
from pff.utils.plots import plot_ind_json

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pytest.importorskip("torch")


def test_plot_ind_json(tmp_path):
    """Plot an individual loaded from a JSON fixture and ensure the file is created."""
    fixture = ROOT / "tests" / "fixtures" / "study_ctx_tgt.json"
    with open(fixture) as f:
        study = json.load(f)
    individual = study["target"][0]

    requested_out = tmp_path / "plot.png"
    saved_out = tmp_path / "plot.png"
    returned_path = plot_ind_json(individual, file_name=str(requested_out))
    assert returned_path == str(saved_out)
    assert saved_out.exists()
