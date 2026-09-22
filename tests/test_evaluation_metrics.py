from evaluation.metrics import bootstrap_mae_ci, regression_metrics


def test_regression_metrics_reference_values():
    metrics = regression_metrics([10.0, 12.0, 14.0], [10.5, 11.0, 15.0])
    assert metrics["n"] == 3
    assert round(metrics["mae_years"], 6) == round(2.5 / 3, 6)
    assert round(metrics["rmse_years"], 6) == round((2.25 / 3) ** 0.5, 6)
    assert metrics["median_absolute_error_years"] == 1.0
    assert round(metrics["mean_bias_years"], 6) == round(0.5 / 3, 6)
    assert metrics["within_0_5_years"] == 1 / 3
    assert metrics["within_1_0_years"] == 1.0


def test_bootstrap_mae_ci_is_deterministic():
    first = bootstrap_mae_ci([1, 2, 3], [1.1, 2.2, 2.7], iterations=100, seed=7)
    second = bootstrap_mae_ci([1, 2, 3], [1.1, 2.2, 2.7], iterations=100, seed=7)
    assert first == second
    assert first[0] <= first[1]
