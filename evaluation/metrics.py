from __future__ import annotations

import math
import random
import statistics
from typing import Iterable


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    truth = [float(x) for x in y_true]
    pred = [float(x) for x in y_pred]

    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not truth:
        raise ValueError("at least one sample is required")

    errors = [p - t for t, p in zip(truth, pred)]
    absolute = [abs(x) for x in errors]
    squared = [x * x for x in errors]
    n = len(truth)

    return {
        "n": n,
        "mae_years": sum(absolute) / n,
        "rmse_years": math.sqrt(sum(squared) / n),
        "median_absolute_error_years": statistics.median(absolute),
        "mean_bias_years": sum(errors) / n,
        "within_0_5_years": sum(x <= 0.5 for x in absolute) / n,
        "within_1_0_years": sum(x <= 1.0 for x in absolute) / n,
    }


def bootstrap_mae_ci(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    iterations: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> tuple[float, float]:
    truth = [float(x) for x in y_true]
    pred = [float(x) for x in y_pred]

    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not truth:
        raise ValueError("at least one sample is required")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    n = len(truth)
    maes: list[float] = []

    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        mae = sum(abs(pred[i] - truth[i]) for i in indices) / n
        maes.append(mae)

    maes.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, min(iterations - 1, int((alpha / 2) * iterations)))
    upper_index = max(0, min(iterations - 1, int((1 - alpha / 2) * iterations) - 1))
    return maes[lower_index], maes[upper_index]
