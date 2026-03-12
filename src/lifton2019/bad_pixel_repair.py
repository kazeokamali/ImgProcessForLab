from typing import Tuple

import numpy as np
from scipy.ndimage import convolve, label, median_filter


def _directional_line_repair(
    image: np.ndarray,
    mask: np.ndarray,
    line_aspect_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    repaired = image.copy()
    remaining = mask.copy()

    labeled, num = label(mask.astype(np.uint8))
    if num <= 0:
        return repaired, remaining

    for comp_id in range(1, num + 1):
        ys, xs = np.where(labeled == comp_id)
        if ys.size == 0:
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        h = y_max - y_min + 1
        w = x_max - x_min + 1

        # Vertical-line-like cluster -> interpolate from left/right (perpendicular).
        if h >= max(2, int(line_aspect_ratio * max(w, 1))):
            for y, x in zip(ys, xs):
                if x <= 0 or x >= repaired.shape[1] - 1:
                    continue
                if (not remaining[y, x - 1]) and (not remaining[y, x + 1]):
                    repaired[y, x] = 0.5 * (repaired[y, x - 1] + repaired[y, x + 1])
                    remaining[y, x] = False
        # Horizontal-line-like cluster -> interpolate from up/down (perpendicular).
        elif w >= max(2, int(line_aspect_ratio * max(h, 1))):
            for y, x in zip(ys, xs):
                if y <= 0 or y >= repaired.shape[0] - 1:
                    continue
                if (not remaining[y - 1, x]) and (not remaining[y + 1, x]):
                    repaired[y, x] = 0.5 * (repaired[y - 1, x] + repaired[y + 1, x])
                    remaining[y, x] = False

    return repaired, remaining


def repair_bad_pixels(
    image: np.ndarray,
    bad_mask: np.ndarray,
    window_size: int = 3,
    max_iterations: int = 6,
    enable_directional_line_repair: bool = True,
    directional_line_aspect_ratio: float = 6.0,
) -> np.ndarray:
    if image.shape != bad_mask.shape:
        raise ValueError(
            f"Bad mask shape mismatch: image {image.shape}, mask {bad_mask.shape}"
        )
    if not np.any(bad_mask):
        return image.astype(np.float32, copy=True)

    repaired = image.astype(np.float32, copy=True)
    remaining = bad_mask.astype(bool, copy=True)

    if enable_directional_line_repair:
        repaired, remaining = _directional_line_repair(
            repaired,
            remaining,
            line_aspect_ratio=float(directional_line_aspect_ratio),
        )

    size = max(3, int(window_size))
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), dtype=np.float32)

    # Iterative neighborhood interpolation for isolated points and small clusters.
    for _ in range(max(1, int(max_iterations))):
        if not np.any(remaining):
            break
        valid = ~remaining
        value_sum = convolve(repaired * valid.astype(np.float32), kernel, mode="mirror")
        valid_count = convolve(valid.astype(np.float32), kernel, mode="mirror")
        fillable = remaining & (valid_count > 0)
        if not np.any(fillable):
            break
        repaired[fillable] = value_sum[fillable] / valid_count[fillable]
        remaining[fillable] = False

    # Fallback for unresolved pixels.
    if np.any(remaining):
        med = median_filter(repaired, size=size, mode="mirror")
        repaired[remaining] = med[remaining]

    return repaired

