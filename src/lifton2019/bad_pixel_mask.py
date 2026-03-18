import os
from typing import Dict, Tuple

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import binary_dilation, label, median_filter

from .models import BadPixelConfig, CalibrationSet


def _robust_outlier_mask(image: np.ndarray, sigma: float, eps: float = 1e-6) -> np.ndarray:
    values = image.ravel().astype(np.float64, copy=False)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    robust_std = max(1.4826 * mad, eps)
    z = np.abs((image.astype(np.float64, copy=False) - med) / robust_std)
    return z > float(sigma)


def _neighbor_deviation_mask(
    image: np.ndarray, neighborhood_size: int, sigma: float
) -> np.ndarray:
    size = max(3, int(neighborhood_size))
    if size % 2 == 0:
        size += 1
    local_median = median_filter(image, size=size, mode="mirror")
    residual = image - local_median
    return _robust_outlier_mask(residual, sigma=sigma)


def _filter_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    min_size = max(1, int(min_component_size))
    if min_size <= 1:
        return mask

    labeled, num = label(mask.astype(np.uint8))
    if num <= 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def _load_known_mask(path: str, shape: Tuple[int, int]) -> np.ndarray:
    if not path:
        return np.zeros(shape, dtype=bool)
    if not os.path.exists(path):
        raise ValueError(f"Known bad pixel mask path does not exist: {path}")

    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    elif lower.endswith(".tif") or lower.endswith(".tiff"):
        arr = iio.imread(path)
    else:
        # Support plain-text list of 0-based or 1-based column indexes.
        cols = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.isdigit():
                    cols.append(int(s))
        mask = np.zeros(shape, dtype=bool)
        if not cols:
            return mask
        # Auto-detect 1-based column list if present.
        one_based = min(cols) >= 1
        width = shape[1]
        for col in cols:
            c = (col - 1) if one_based else col
            if 0 <= c < width:
                mask[:, c] = True
        return mask

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(
            f"Known bad mask shape mismatch: expected {shape}, got {arr.shape}"
        )
    return arr > 0


def build_bad_pixel_mask(
    calibration: CalibrationSet,
    config: BadPixelConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    shape = calibration.dark_avg.shape

    if not config.enabled:
        empty = np.zeros(shape, dtype=bool)
        return empty, {"total": 0}

    masks = []
    stats: Dict[str, int] = {}

    if config.enable_flat_neighbor_check:
        flat_stack = np.stack(calibration.flat_avgs, axis=0) - calibration.dark_avg
        flat_mean = np.mean(flat_stack, axis=0).astype(np.float32, copy=False)
        mask_flat = _neighbor_deviation_mask(
            flat_mean,
            neighborhood_size=config.neighborhood_size,
            sigma=config.flat_neighbor_sigma,
        )
        masks.append(mask_flat)
        stats["flat_neighbor"] = int(mask_flat.sum())

    if config.enable_dark_neighbor_check:
        dark_mean = calibration.dark_avg.astype(np.float32, copy=False)
        mask_dark = _neighbor_deviation_mask(
            dark_mean,
            neighborhood_size=config.neighborhood_size,
            sigma=config.dark_neighbor_sigma,
        )
        masks.append(mask_dark)
        stats["dark_neighbor"] = int(mask_dark.sum())

    if config.enable_stability_check:
        stability_maps = []
        if calibration.dark_std is not None:
            stability_maps.append(calibration.dark_std)
        if calibration.flat_stds:
            stability_maps.extend(calibration.flat_stds)

        if stability_maps:
            stability = np.mean(np.stack(stability_maps, axis=0), axis=0)
            mask_stability = _robust_outlier_mask(
                stability.astype(np.float32, copy=False),
                sigma=config.stability_sigma,
            )
            masks.append(mask_stability)
            stats["stability"] = int(mask_stability.sum())
        else:
            stats["stability"] = 0

    known_union = np.zeros(shape, dtype=bool)
    if config.known_mask_path:
        known_union |= _load_known_mask(config.known_mask_path, shape)
    if config.known_badline_path:
        known_union |= _load_known_mask(config.known_badline_path, shape)
    if np.any(known_union):
        masks.append(known_union)
    stats["known"] = int(known_union.sum())

    if masks:
        merged = np.logical_or.reduce(masks)
    else:
        merged = np.zeros(shape, dtype=bool)

    merged = _filter_small_components(merged, config.min_component_size)

    if config.dilation_radius > 0:
        iterations = max(1, int(config.dilation_radius))
        merged = binary_dilation(merged, iterations=iterations)

    stats["total"] = int(merged.sum())
    return merged.astype(bool), stats
