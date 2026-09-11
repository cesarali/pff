import pandas as pd

from pff.utils.experiments_potsprocessing import (
    available_tasks_and_checkpoints_new,
    metrics_list_to_pandas_new,
)


def test_metrics_list_to_pandas_new_extracts_per_drug_metrics():
    metrics_list = [
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/repaglinide/log_r2_std"
            ),
            "metricValue": "1.479039340900015",
            "timestamp": 1774963885753,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/caffeine (137X)/log_r2_std"
            ),
            "metricValue": "0.35768113430803637",
            "timestamp": 1774963885756,
            "step": 6700,
            "epoch": 99,
        },
    ]

    df = metrics_list_to_pandas_new(
        metrics_list=metrics_list,
        model_name="FlowPK",
        metric_name="log_r2_std",
        epoch="last",
        log_prefix="Empirical",
        task_name="empirical/predictive_metrics/lenuzza-2016",
        checkpoint_name="log_rmse",
    )

    expected = pd.DataFrame(
        {
            "drug": ["caffeine", "repaglinide"],
            "FlowPK": [0.35768113430803637, 1.479039340900015],
        }
    )

    pd.testing.assert_frame_equal(df, expected)


def test_metrics_list_to_pandas_new_ignores_summary_metrics_and_keeps_latest_timestamp():
    metrics_list = [
        {
            "metricName": "Empirical/empirical/summary__end/log_rmse",
            "metricValue": "1.7550567366182803",
            "timestamp": 1774963678938,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/caffeine (137X)/rmse_std"
            ),
            "metricValue": "0.00035768113430803637",
            "timestamp": 1774963885756,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/caffeine (137X)/rmse_std"
            ),
            "metricValue": "0.123",
            "timestamp": 1774963885000,
            "step": 6700,
            "epoch": 99,
        },
    ]

    df = metrics_list_to_pandas_new(
        metrics_list=metrics_list,
        model_name="FlowPK",
        metric_name="rmse_std",
        epoch=99,
        log_prefix="Empirical",
        task_name="empirical/predictive_metrics/lenuzza-2016",
        checkpoint_name="log_rmse",
    )

    expected = pd.DataFrame(
        {
            "drug": ["caffeine"],
            "FlowPK": [0.00035768113430803637],
        }
    )

    pd.testing.assert_frame_equal(df, expected)


def test_metrics_list_to_pandas_new_filters_checkpoint_suffix():
    metrics_list = [
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__last/repaglinide/log_rmse"
            ),
            "metricValue": "0.9",
            "timestamp": 1774963885751,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__end/repaglinide/log_rmse"
            ),
            "metricValue": "0.8",
            "timestamp": 1774963885752,
            "step": 6700,
            "epoch": 99,
        },
    ]

    df = metrics_list_to_pandas_new(
        metrics_list=metrics_list,
        model_name="FlowPK",
        metric_name="log_rmse",
        epoch=99,
        log_prefix="Empirical",
        task_name="empirical/predictive_metrics/lenuzza-2016",
        checkpoint_name="end",
    )

    expected = pd.DataFrame(
        {
            "drug": ["repaglinide"],
            "FlowPK": [0.8],
        }
    )

    pd.testing.assert_frame_equal(df, expected)


def test_metrics_list_to_pandas_new_accepts_epoch_end_as_alias_for_last():
    metrics_list = [
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/repaglinide/log_rmse"
            ),
            "metricValue": "0.8",
            "timestamp": 1774963885752,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__log_rmse/repaglinide/log_rmse"
            ),
            "metricValue": "0.9",
            "timestamp": 1774963885751,
            "step": 6600,
            "epoch": 98,
        },
    ]

    df = metrics_list_to_pandas_new(
        metrics_list=metrics_list,
        model_name="AICMET",
        metric_name="log_rmse",
        epoch="end",
        log_prefix="Empirical",
        task_name="empirical/predictive_metrics/lenuzza-2016",
        checkpoint_name="log_rmse",
    )

    expected = pd.DataFrame(
        {
            "drug": ["repaglinide"],
            "AICMET": [0.8],
        }
    )

    pd.testing.assert_frame_equal(df, expected)


def test_available_tasks_and_checkpoints_new_summarizes_new_metric_layout():
    metrics_list = [
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__end/repaglinide/log_rmse"
            ),
            "metricValue": "0.8",
            "timestamp": 1774963885752,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": (
                "Empirical/empirical/predictive_metrics/"
                "lenuzza-2016__end/caffeine (137X)/log_rmse"
            ),
            "metricValue": "0.6",
            "timestamp": 1774963885753,
            "step": 6700,
            "epoch": 99,
        },
        {
            "metricName": "Empirical/empirical/summary__end/log_rmse",
            "metricValue": "1.7",
            "timestamp": 1774963678938,
            "step": 6700,
            "epoch": 99,
        },
    ]

    df = available_tasks_and_checkpoints_new(
        metrics_list=metrics_list,
        epoch=99,
        log_prefix="Empirical",
    )

    predictive_row = df[
        (df["task_name"] == "empirical/predictive_metrics/lenuzza-2016")
        & (df["checkpoint_name"] == "end")
        & (df["metric_name"] == "log_rmse")
    ].iloc[0]
    assert predictive_row["is_per_drug"]
    assert predictive_row["n_drugs"] == 2

    summary_row = df[
        (df["task_name"] == "empirical/summary")
        & (df["checkpoint_name"] == "end")
        & (df["metric_name"] == "log_rmse")
    ].iloc[0]
    assert not summary_row["is_per_drug"]
    assert summary_row["n_drugs"] == 0
