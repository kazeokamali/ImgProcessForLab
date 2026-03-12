import os
import re
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np

from .models import CalibrationSet, ProcessingConfig


def _natural_key(text: str):
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def collect_image_files(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []

    valid_exts = (".tif", ".tiff", ".raw")
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and name.lower().endswith(valid_exts)
    ]
    files.sort(key=lambda p: _natural_key(os.path.basename(p)))
    return files


def load_image(path: str, raw_shape: Tuple[int, int]) -> np.ndarray:
    if path.lower().endswith(".raw"):
        height, width = raw_shape
        data = np.fromfile(path, dtype=np.uint16)
        expected = height * width
        if data.size != expected:
            raise ValueError(
                f"RAW size mismatch for {path}: expected {width}x{height}={expected}, got {data.size}"
            )
        return data.reshape((height, width)).astype(np.float32, copy=False)

    image = iio.imread(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.asarray(image, dtype=np.float32)


def average_images(paths: List[str], raw_shape: Tuple[int, int]) -> np.ndarray:
    if not paths:
        raise ValueError("No image files provided for averaging.")

    acc = None
    for path in paths:
        img = load_image(path, raw_shape)
        if acc is None:
            acc = np.zeros_like(img, dtype=np.float64)
        if img.shape != acc.shape:
            raise ValueError(f"Shape mismatch in averaging: {path} has shape {img.shape}")
        acc += img.astype(np.float64, copy=False)

    avg = acc / float(len(paths))
    return avg.astype(np.float32, copy=False)


def mean_std_images(paths: List[str], raw_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No image files provided for mean/std computation.")

    mean = None
    m2 = None
    count = 0
    for path in paths:
        x = load_image(path, raw_shape).astype(np.float64, copy=False)
        if mean is None:
            mean = np.zeros_like(x, dtype=np.float64)
            m2 = np.zeros_like(x, dtype=np.float64)
        if x.shape != mean.shape:
            raise ValueError(f"Shape mismatch in mean/std: {path} has shape {x.shape}")

        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2

    if count <= 1:
        std = np.zeros_like(mean, dtype=np.float64)
    else:
        std = np.sqrt(np.maximum(m2 / (count - 1), 0.0))
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def _extract_point_key(filename: str, pattern: re.Pattern) -> Optional[str]:
    match = pattern.search(filename)
    if not match:
        return None
    groups = [g for g in match.groups() if g is not None]
    if groups:
        return groups[0]
    return match.group(0)


def resolve_flat_groups(flat_folder: str, num_points: int, point_pattern: str) -> List[Tuple[str, List[str]]]:
    if not os.path.isdir(flat_folder):
        raise ValueError(f"Flat folder does not exist: {flat_folder}")

    subdirs = [
        os.path.join(flat_folder, name)
        for name in os.listdir(flat_folder)
        if os.path.isdir(os.path.join(flat_folder, name))
    ]
    subdirs.sort(key=lambda p: _natural_key(os.path.basename(p)))

    # Preferred layout: one subfolder per point.
    if subdirs:
        groups = []
        for subdir in subdirs:
            files = collect_image_files(subdir)
            if files:
                point_id = os.path.basename(subdir)
                groups.append((point_id, files))
        if len(groups) < num_points:
            raise ValueError(
                f"Found only {len(groups)} flat point groups in subfolders, expected at least {num_points}."
            )
        return groups[:num_points]

    # Fallback layout: all point files in one folder; group by regex.
    all_files = collect_image_files(flat_folder)
    if not all_files:
        raise ValueError(f"No flat files found: {flat_folder}")

    try:
        compiled = re.compile(point_pattern)
    except re.error as e:
        raise ValueError(f"Invalid point regex pattern: {e}") from e

    grouped: Dict[str, List[str]] = {}
    for path in all_files:
        name = os.path.basename(path)
        key = _extract_point_key(name, compiled)
        if key is None:
            continue
        grouped.setdefault(key, []).append(path)

    if len(grouped) < num_points:
        raise ValueError(
            f"Resolved only {len(grouped)} point groups from flat files, expected at least {num_points}."
        )

    sorted_keys = sorted(grouped.keys(), key=_natural_key)
    result = []
    for key in sorted_keys[:num_points]:
        files = sorted(grouped[key], key=lambda p: _natural_key(os.path.basename(p)))
        result.append((key, files))
    return result


def _compute_reference_value(
    image: np.ndarray, use_roi_reference: bool, roi: Optional[Tuple[int, int, int, int]]
) -> float:
    if not use_roi_reference or roi is None:
        return float(np.mean(image))

    x, y, w, h = roi
    h_img, w_img = image.shape
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w_img, x0 + max(1, int(w)))
    y1 = min(h_img, y0 + max(1, int(h)))
    if x1 <= x0 or y1 <= y0:
        return float(np.mean(image))
    return float(np.mean(image[y0:y1, x0:x1]))


def load_calibration_set(
    *,
    dark_folder: str,
    flat_folder: str,
    config: ProcessingConfig,
    compute_std_maps: bool = False,
) -> CalibrationSet:
    raw_shape = (config.raw_height, config.raw_width)

    dark_files = collect_image_files(dark_folder)
    if not dark_files:
        raise ValueError(f"No dark files found: {dark_folder}")
    dark_std = None
    if compute_std_maps:
        dark_avg, dark_std = mean_std_images(dark_files, raw_shape)
    else:
        dark_avg = average_images(dark_files, raw_shape)

    groups = resolve_flat_groups(flat_folder, config.num_points, config.point_pattern)

    flat_avgs: List[np.ndarray] = []
    flat_refs: List[float] = []
    point_ids: List[str] = []
    frame_counts: Dict[str, int] = {}
    flat_stds: List[np.ndarray] = []

    for point_id, files in groups:
        flat_std = None
        if compute_std_maps:
            flat_avg, flat_std = mean_std_images(files, raw_shape)
        else:
            flat_avg = average_images(files, raw_shape)
        if flat_avg.shape != dark_avg.shape:
            raise ValueError(
                f"Shape mismatch between dark and flat point '{point_id}': "
                f"{dark_avg.shape} vs {flat_avg.shape}"
            )

        flat_dark = flat_avg - dark_avg
        reference = _compute_reference_value(
            flat_dark, config.use_roi_reference, config.reference_roi
        )

        flat_avgs.append(flat_avg.astype(np.float32, copy=False))
        flat_refs.append(reference)
        point_ids.append(str(point_id))
        frame_counts[str(point_id)] = len(files)
        if flat_std is not None:
            flat_stds.append(flat_std.astype(np.float32, copy=False))

    return CalibrationSet(
        dark_avg=dark_avg.astype(np.float32, copy=False),
        flat_avgs=flat_avgs,
        flat_refs=np.asarray(flat_refs, dtype=np.float32),
        point_ids=point_ids,
        frame_counts=frame_counts,
        dark_std=dark_std.astype(np.float32, copy=False) if dark_std is not None else None,
        flat_stds=flat_stds if flat_stds else None,
    )
