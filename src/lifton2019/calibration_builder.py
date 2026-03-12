from typing import List

import numpy as np

from .models import CalibrationSet, PiecewiseKnots


def build_piecewise_knots(calibration: CalibrationSet) -> PiecewiseKnots:
    if len(calibration.flat_avgs) < 2:
        raise ValueError("At least 2 flat points are required for piecewise linear correction.")

    x_knots = np.stack(
        [flat_avg - calibration.dark_avg for flat_avg in calibration.flat_avgs],
        axis=0,
    ).astype(np.float32, copy=False)
    y_knots = calibration.flat_refs.astype(np.float32, copy=False)

    # Sort points by reference intensity.
    order = np.argsort(y_knots)
    x_knots = x_knots[order]
    y_knots = y_knots[order]
    point_ids: List[str] = [calibration.point_ids[int(i)] for i in order]

    # Enforce non-decreasing x along point axis to make interval selection stable.
    x_knots = np.maximum.accumulate(x_knots, axis=0)
    return PiecewiseKnots(x_knots=x_knots, y_knots=y_knots, point_ids=point_ids)


def apply_piecewise_linear_correction(
    projection_dark_corrected: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if x_knots.ndim != 3:
        raise ValueError("x_knots must be [N, H, W].")
    if y_knots.ndim != 1:
        raise ValueError("y_knots must be [N].")
    if x_knots.shape[0] != y_knots.shape[0]:
        raise ValueError("Point count mismatch between x_knots and y_knots.")

    n_points, h, w = x_knots.shape
    if projection_dark_corrected.shape != (h, w):
        raise ValueError(
            f"Projection shape mismatch: expected {(h, w)}, got {projection_dark_corrected.shape}"
        )
    if n_points < 2:
        raise ValueError("At least 2 points are required for piecewise correction.")

    p = projection_dark_corrected.astype(np.float32, copy=False)
    corrected = np.empty_like(p, dtype=np.float32)
    assigned = np.zeros_like(p, dtype=bool)

    for i in range(n_points - 1):
        x0 = x_knots[i]
        x1 = x_knots[i + 1]
        y0 = float(y_knots[i])
        y1 = float(y_knots[i + 1])

        denom = x1 - x0
        denom_safe = np.where(np.abs(denom) < epsilon, np.sign(denom) * epsilon, denom)
        denom_safe = np.where(denom_safe == 0.0, epsilon, denom_safe)

        # Pixel-wise slope for this segment.
        slope = (y1 - y0) / denom_safe
        value = y0 + slope * (p - x0)

        if i == 0:
            mask = p <= x1
        elif i == (n_points - 2):
            mask = p >= x0
        else:
            mask = (p >= x0) & (p <= x1)

        write_mask = mask & (~assigned)
        corrected[write_mask] = value[write_mask]
        assigned[write_mask] = True

    # Fallback for pathological data.
    corrected[~assigned] = p[~assigned]
    return corrected

