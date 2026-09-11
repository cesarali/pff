import pathlib
import sys

import matplotlib.pyplot as plt
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pytest.importorskip("torch")

from pff.utils.plots import plot_study_json_with_dosing_histograms


def _study_with_mixed_dosing_routes() -> dict:
    """Return a small StudyJSON fixture with multiple dosing routes and values."""

    return {
        "context": [
            {
                "name_id": "C01",
                "observations": [1.0, 2.0],
                "observation_times": [0.5, 1.0],
                "dosing": [10.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            },
            {
                "name_id": "C02",
                "observations": [1.2, 1.8],
                "observation_times": [0.5, 1.0],
                "dosing": [20.0],
                "dosing_type": ["iv"],
                "dosing_times": [0.0],
                "dosing_name": ["iv"],
            },
        ],
        "target": [
            {
                "name_id": "T01",
                "observations": [0.9, 1.4],
                "observation_times": [0.5, 1.0],
                "remaining": [1.1, 0.8],
                "remaining_times": [1.5, 2.0],
                "dosing": [15.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
                "prediction_samples": [[0.95, 0.8], [1.05, 0.9]],
                "prediction_times": [1.5, 2.0],
            },
            {
                "name_id": "T02",
                "observations": [1.1, 1.6],
                "observation_times": [0.5, 1.0],
                "remaining": [0.9, 0.7],
                "remaining_times": [1.5, 2.0],
                "dosing": [25.0],
                "dosing_type": ["iv"],
                "dosing_times": [0.0],
                "dosing_name": ["iv"],
            },
            {
                "name_id": "T03",
                "observations": [0.8, 1.1],
                "observation_times": [0.5, 1.0],
                "remaining": [0.7, 0.5],
                "remaining_times": [1.5, 2.0],
                "dosing": [30.0],
                "dosing_type": ["sc"],
                "dosing_times": [0.0],
                "dosing_name": ["sc"],
            },
        ],
        "meta_data": {
            "study_name": "MixedRoutes",
            "substance_name": "DrugY",
        },
    }


def test_plot_study_json_with_dosing_histograms(tmp_path) -> None:
    study = _study_with_mixed_dosing_routes()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    requested_out = tmp_path / "study_dosing_histograms.png"
    saved_out = tmp_path / "study_dosing_histograms.png"
    returned_path = plot_study_json_with_dosing_histograms(
        study,
        axes=axes,
        file_name=str(requested_out),
        log_scale=False,
    )

    fig.canvas.draw()

    assert returned_path == str(saved_out)
    assert saved_out.exists()
    assert axes[1].get_title() == "Dosing Route Counts"
    assert [patch.get_height() for patch in axes[1].patches] == [2, 2, 1]
    assert axes[2].get_title() == "Dose Values by Route"
    assert axes[2].get_legend() is not None

    plt.close(fig)
