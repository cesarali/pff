import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pytest.importorskip("torch")

from pff.utils.plots import plot_study_json


def test_plot_study_json(tmp_path):
    #fixture = ROOT / "tests" / "data_empirical" / "fixtures" / "study_ctx_tgt.json"
    fixture = ROOT / "tests" / "fixtures" / "study_ctx_tgt.json"
    with open(fixture) as f:
        study = json.load(f)

    requested_out = tmp_path / "plot.png"
    saved_out = tmp_path / "plot.png"
    returned_path = plot_study_json(study, file_name=str(requested_out))
    assert returned_path == str(saved_out)
    assert saved_out.exists()
