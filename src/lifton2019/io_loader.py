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


def load_single_root_calibration_set(
    *,
    calibration_root: str,
    config: ProcessingConfig,
    compute_std_maps: bool = False,
) -> CalibrationSet:
    if not calibration_root or not os.path.isdir(calibration_root):
        raise ValueError(f"单次标定目录不存在: {calibration_root}")

    dark_default = os.path.join(calibration_root, "dark")
    flat_default = os.path.join(calibration_root, "flat")
    if os.path.isdir(dark_default) and os.path.isdir(flat_default):
        return load_calibration_set(
            dark_folder=dark_default,
            flat_folder=flat_default,
            config=config,
            compute_std_maps=compute_std_maps,
        )

    subdirs = [
        os.path.join(calibration_root, name)
        for name in os.listdir(calibration_root)
        if os.path.isdir(os.path.join(calibration_root, name))
    ]
    subdirs.sort(key=lambda p: _natural_key(os.path.basename(p)))
    if not subdirs:
        raise ValueError("单次标定目录下未找到任何子目录。")

    dark_name_tokens = ("dark", "black", "暗", "黑")
    dark_candidates = []
    for subdir in subdirs:
        name = os.path.basename(subdir).lower()
        files = collect_image_files(subdir)
        if files and any(tok in name for tok in dark_name_tokens):
            dark_candidates.append(subdir)

    if len(dark_candidates) != 1:
        raise ValueError(
            "单次标定目录解析失败：请保证子目录中有且仅有一个 dark 目录"
            "（目录名建议包含 dark/black/暗/黑）。"
        )

    dark_folder = dark_candidates[0]
    dark_files = collect_image_files(dark_folder)
    if not dark_files:
        raise ValueError(f"dark 子目录中未找到图像文件: {dark_folder}")

    raw_shape = (config.raw_height, config.raw_width)
    dark_std = None
    if compute_std_maps:
        dark_avg, dark_std = mean_std_images(dark_files, raw_shape)
    else:
        dark_avg = average_images(dark_files, raw_shape)

    flat_groups: List[Tuple[str, List[str]]] = []
    for subdir in subdirs:
        if subdir == dark_folder:
            continue
        files = collect_image_files(subdir)
        if files:
            flat_groups.append((os.path.basename(subdir), files))

    if len(flat_groups) < config.num_points:
        raise ValueError(
            f"单次标定目录中仅找到 {len(flat_groups)} 组 flat 子目录，"
            f"少于设置点数 {config.num_points}。"
        )

    flat_groups = flat_groups[: int(config.num_points)]

    flat_avgs: List[np.ndarray] = []
    flat_refs: List[float] = []
    point_ids: List[str] = []
    frame_counts: Dict[str, int] = {}
    flat_stds: List[np.ndarray] = []

    for point_id, files in flat_groups:
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


def load_paired_subfolder_calibration_set(
    *,
    dark_root_folder: str,
    flat_root_folder: str,
    config: ProcessingConfig,
) -> CalibrationSet:
    point_ids, dark_avgs, flat_avgs, frame_counts = load_paired_subfolder_point_averages(
        dark_root_folder=dark_root_folder,
        flat_root_folder=flat_root_folder,
        config=config,
    )

    flat_dark_avgs: List[np.ndarray] = []
    flat_refs: List[float] = []
    for point_id, dark_avg, flat_avg in zip(point_ids, dark_avgs, flat_avgs):
        if dark_avg.shape != flat_avg.shape:
            raise ValueError(
                f"点位 {point_id} 尺寸不一致: dark={dark_avg.shape}, flat={flat_avg.shape}"
            )
        flat_dark = (flat_avg - dark_avg).astype(np.float32, copy=False)
        ref_value = _compute_reference_value(
            flat_dark, config.use_roi_reference, config.reference_roi
        )
        flat_dark_avgs.append(flat_dark)
        flat_refs.append(ref_value)

    dark_zero = np.zeros_like(flat_dark_avgs[0], dtype=np.float32)
    return CalibrationSet(
        dark_avg=dark_zero,
        flat_avgs=flat_dark_avgs,
        flat_refs=np.asarray(flat_refs, dtype=np.float32),
        point_ids=point_ids,
        frame_counts=frame_counts,
        dark_std=None,
        flat_stds=None,
    )


def load_paired_subfolder_point_averages(
    *,
    dark_root_folder: str,
    flat_root_folder: str,
    config: ProcessingConfig,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    if not dark_root_folder or not os.path.isdir(dark_root_folder):
        raise ValueError(f"Dark 根目录不存在: {dark_root_folder}")
    if not flat_root_folder or not os.path.isdir(flat_root_folder):
        raise ValueError(f"Flat 根目录不存在: {flat_root_folder}")

    try:
        compiled = re.compile(config.point_pattern)
    except re.error as e:
        raise ValueError(f"点位正则无效: {e}") from e

    raw_shape = (config.raw_height, config.raw_width)

    def collect_groups(root_folder: str) -> Dict[str, Tuple[str, List[str]]]:
        result: Dict[str, Tuple[str, List[str]]] = {}
        subdirs = [
            os.path.join(root_folder, name)
            for name in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, name))
        ]
        subdirs.sort(key=lambda p: _natural_key(os.path.basename(p)))
        for subdir in subdirs:
            files = collect_image_files(subdir)
            if not files:
                continue
            folder_name = os.path.basename(subdir)
            key = _extract_point_key(folder_name, compiled)
            if key is None:
                continue
            if key in result:
                raise ValueError(f"点位键重复: {key}（目录: {folder_name}）")
            result[key] = (folder_name, files)
        return result

    dark_groups = collect_groups(dark_root_folder)
    flat_groups = collect_groups(flat_root_folder)
    common_keys = sorted(set(dark_groups.keys()) & set(flat_groups.keys()), key=_natural_key)
    if len(common_keys) < config.num_points:
        raise ValueError(
            f"可匹配点位仅 {len(common_keys)} 个，少于设置点数 {config.num_points}。"
        )

    selected_keys = common_keys[: int(config.num_points)]

    point_ids: List[str] = []
    dark_avgs: List[np.ndarray] = []
    flat_avgs: List[np.ndarray] = []
    frame_counts: Dict[str, int] = {}

    for key in selected_keys:
        _dark_name, dark_files = dark_groups[key]
        _flat_name, flat_files = flat_groups[key]

        dark_avg = average_images(dark_files, raw_shape)
        flat_avg = average_images(flat_files, raw_shape)
        if dark_avg.shape != flat_avg.shape:
            raise ValueError(
                f"点位 {key} 尺寸不一致: dark={dark_avg.shape}, flat={flat_avg.shape}"
            )

        point_ids.append(str(key))
        dark_avgs.append(dark_avg.astype(np.float32, copy=False))
        flat_avgs.append(flat_avg.astype(np.float32, copy=False))
        frame_counts[str(key)] = len(dark_files) + len(flat_files)

    return point_ids, dark_avgs, flat_avgs, frame_counts


def load_bad_pixel_calibration_set(
    *,
    dark_folder: str,
    flat_folder: str,
    raw_shape: Tuple[int, int],
    compute_std_maps: bool = False,
) -> CalibrationSet:
    dark_files = collect_image_files(dark_folder)
    if not dark_files:
        raise ValueError(f"坏点流程未找到 dark 文件: {dark_folder}")

    flat_files = collect_image_files(flat_folder)
    if not flat_files:
        raise ValueError(f"坏点流程未找到 flat 文件: {flat_folder}")

    dark_std = None
    if compute_std_maps:
        dark_avg, dark_std = mean_std_images(dark_files, raw_shape)
    else:
        dark_avg = average_images(dark_files, raw_shape)

    flat_std = None
    if compute_std_maps:
        flat_avg, flat_std = mean_std_images(flat_files, raw_shape)
    else:
        flat_avg = average_images(flat_files, raw_shape)

    if flat_avg.shape != dark_avg.shape:
        raise ValueError(
            f"坏点流程中 dark/flat 尺寸不一致: {dark_avg.shape} vs {flat_avg.shape}"
        )

    flat_dark = flat_avg - dark_avg
    reference = float(np.mean(flat_dark))

    return CalibrationSet(
        dark_avg=dark_avg.astype(np.float32, copy=False),
        flat_avgs=[flat_avg.astype(np.float32, copy=False)],
        flat_refs=np.asarray([reference], dtype=np.float32),
        point_ids=["single"],
        frame_counts={"dark": len(dark_files), "flat": len(flat_files)},
        dark_std=dark_std.astype(np.float32, copy=False) if dark_std is not None else None,
        flat_stds=[flat_std.astype(np.float32, copy=False)] if flat_std is not None else None,
    )
