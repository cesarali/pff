import json

from pff import project_dir
from pff.data.data_generation.study_population_stats import (
    BasicObservationStats,
)


# Apply the function and look at the output
def test_study_population_stats_oral():
    fixture = project_dir / "tests" / "fixtures" / "studies_for_stats_calculation_oral.json"

    with open(fixture) as f:
        study = json.load(f)

    stats_obj = BasicObservationStats()
    stats = stats_obj.compute_study_population_statistics(study)

    assert stats['nAUC_mean_percentiles']['P50'] == 4.0
    assert stats['nAUC_sd_percentiles']['P50'] == 0.0
    assert stats['nCmax_mean_percentiles']['P50'] == 2.0
    assert stats['nCmax_sd_percentiles']['P50'] == 0.0
    assert stats['Tmax_mean_percentiles']['P50'] == 1.0
    assert stats['Tmax_sd_percentiles']['P50'] == 0.0
    assert stats['Nobs_mean_percentiles']['P50'] == 4.0
    assert stats['Nobs_total_percentiles']['P50'] == 8.0
    assert stats['Duration_max_percentiles']['P50'] == 4.0
    assert stats['Nstudy'] == 2

def test_study_population_stats_ivbolus():
    fixture = project_dir / "tests" / "fixtures" / "studies_for_stats_calculation_ivbolus.json"

    with open(fixture) as f:
        study = json.load(f)

    stats_obj = BasicObservationStats()
    stats = stats_obj.compute_study_population_statistics(study)

    assert stats['nAUC_mean_percentiles']['P50'] == 6.5
    assert stats['nAUC_sd_percentiles']['P50'] == 0.0
    assert stats['nCmax_mean_percentiles']['P50'] == 4.0
    assert stats['nCmax_sd_percentiles']['P50'] == 0.0
    assert stats['Tmax_mean_percentiles']['P50'] == 0.5
    assert stats['Tmax_sd_percentiles']['P50'] == 0.0
    assert stats['Nobs_mean_percentiles']['P50'] == 4.0
    assert stats['Nobs_total_percentiles']['P50'] == 8.0
    assert stats['Duration_max_percentiles']['P50'] == 4.0
    assert stats['Nstudy'] == 2

if __name__ == "__main__":
    test_study_population_stats_ivbolus()
