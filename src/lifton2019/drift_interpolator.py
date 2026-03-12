import numpy as np

from .models import PiecewiseKnots


def interpolate_array(before: np.ndarray, after: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - alpha) * before + alpha * after).astype(np.float32, copy=False)


def interpolate_knots(before: PiecewiseKnots, after: PiecewiseKnots, alpha: float) -> PiecewiseKnots:
    if before.x_knots.shape != after.x_knots.shape:
        raise ValueError(
            f"x_knots shape mismatch: {before.x_knots.shape} vs {after.x_knots.shape}"
        )
    if before.y_knots.shape != after.y_knots.shape:
        raise ValueError(
            f"y_knots shape mismatch: {before.y_knots.shape} vs {after.y_knots.shape}"
        )
    if before.point_ids != after.point_ids:
        raise ValueError("Point IDs mismatch between before and after calibration sets.")

    x = interpolate_array(before.x_knots, after.x_knots, alpha)
    # Preserve monotonicity after interpolation.
    x = np.maximum.accumulate(x, axis=0)
    y = interpolate_array(before.y_knots, after.y_knots, alpha)
    return PiecewiseKnots(x_knots=x, y_knots=y, point_ids=list(before.point_ids))

