from structural_analytics.metrics import regression_metrics


def test_perfect_prediction():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
