import json
import os
from typing import Callable, Optional, Tuple

import numpy as np
import tifffile as tiff

from .calibration_builder import apply_piecewise_linear_correction, build_piecewise_knots
from .drift_interpolator import interpolate_array, interpolate_knots
from .io_loader import collect_image_files, load_calibration_set, load_image
from .metrics import MetricsCollector, compute_air_roi_mean, compute_stripe_strength
from .models import CalibrationSet, ProcessingConfig, ProcessingResult


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _normalize_roi(roi: Tuple[int, int, int, int], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = shape
    x, y, rw, rh = [int(v) for v in roi]
    x0 = max(0, min(w - 1, x))
    y0 = max(0, min(h - 1, y))
    x1 = max(x0 + 1, min(w, x0 + max(1, rw)))
    y1 = max(y0 + 1, min(h, y0 + max(1, rh)))
    return x0, y0, x1, y1


def _roi_mean(image: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = _normalize_roi(roi, image.shape)
    return float(np.mean(image[y0:y1, x0:x1]))


def _roi_column_profile(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = _normalize_roi(roi, image.shape)
    return np.mean(image[y0:y1, x0:x1], axis=0).astype(np.float32, copy=False)


def _estimate_profile_blend_coeff(
    query_profile: np.ndarray,
    profile0: np.ndarray,
    profile1: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Estimate single global blend coefficient c for:
      query_profile ~= c * profile0 + (1-c) * profile1
    """
    q = np.asarray(query_profile, dtype=np.float32).reshape(-1)
    p0 = np.asarray(profile0, dtype=np.float32).reshape(-1)
    p1 = np.asarray(profile1, dtype=np.float32).reshape(-1)
    if q.shape[0] != p0.shape[0] or q.shape[0] != p1.shape[0]:
        raise ValueError(
            f"ROI width mismatch: query={q.shape[0]}, profile0={p0.shape[0]}, profile1={p1.shape[0]}"
        )

    d = p0 - p1
    denom = float(np.dot(d, d))
    if denom < epsilon:
        err0 = float(np.mean((q - p0) ** 2))
        err1 = float(np.mean((q - p1) ** 2))
        return 1.0 if err0 <= err1 else 0.0

    # q ~= c * p0 + (1-c) * p1 => q - p1 ~= c * (p0 - p1)
    c = float(np.dot(q - p1, d) / denom)
    c = float(np.clip(c, 0.0, 1.0))
    return c


def _select_bracketing_pair(
    sorted_values: np.ndarray,
    query_value: float,
) -> Tuple[int, int]:
    values = np.asarray(sorted_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("sorted_values is empty.")
    if values.size == 1:
        return 0, 0

    q = float(query_value)
    if q <= float(values[0]):
        return 0, 1
    if q >= float(values[-1]):
        return values.size - 2, values.size - 1

    right = int(np.searchsorted(values, q, side="right"))
    left = right - 1
    return left, right


def _interpolate_stack_by_scalar(
    stack: np.ndarray,
    axis_values: np.ndarray,
    query: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    values = np.asarray(axis_values, dtype=np.float64).reshape(-1)
    if stack.shape[0] != values.shape[0]:
        raise ValueError(
            f"插值轴长度不匹配: stack={stack.shape[0]}, axis_values={values.shape[0]}"
        )

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_stack = stack[order]

    q = float(query)
    if q <= sorted_values[0]:
        return sorted_stack[0].astype(np.float32, copy=False)
    if q >= sorted_values[-1]:
        return sorted_stack[-1].astype(np.float32, copy=False)

    right = int(np.searchsorted(sorted_values, q, side="right"))
    left = right - 1
    v0 = float(sorted_values[left])
    v1 = float(sorted_values[right])
    if abs(v1 - v0) < epsilon:
        alpha = 0.0
    else:
        alpha = (q - v0) / (v1 - v0)

    out = (1.0 - alpha) * sorted_stack[left] + alpha * sorted_stack[right]
    return out.astype(np.float32, copy=False)


def run_lifton2019_pipeline(
    *,
    projection_folder: str,
    dark_before_folder: Optional[str] = None,
    flat_before_folder: Optional[str] = None,
    dark_after_folder: Optional[str] = None,
    flat_after_folder: Optional[str] = None,
    output_folder: str,
    config: ProcessingConfig,
    calib_before: Optional[CalibrationSet] = None,
    calib_after: Optional[CalibrationSet] = None,
    progress_callback: ProgressCallback = None,
) -> ProcessingResult:
    """Run pure Lifton-like correction flow without bad-pixel pre-processing."""
    if config.num_points < 2:
        raise ValueError("num_points must be >= 2.")

    os.makedirs(output_folder, exist_ok=True)
    output_projection_dir = os.path.join(output_folder, "projections_corrected")
    os.makedirs(output_projection_dir, exist_ok=True)

    projection_paths = collect_image_files(projection_folder)
    if not projection_paths:
        raise ValueError(f"No projection files found: {projection_folder}")

    raw_shape: Tuple[int, int] = (config.raw_height, config.raw_width)

    if calib_before is None:
        if not dark_before_folder or not flat_before_folder:
            raise ValueError("缺少扫描前 dark/flat 目录。")
        if progress_callback:
            progress_callback(0, len(projection_paths), "正在加载标定数据...")
        calib_before = load_calibration_set(
            dark_folder=dark_before_folder,
            flat_folder=flat_before_folder,
            config=config,
        )

    if calib_after is None:
        if dark_after_folder or flat_after_folder:
            if not dark_after_folder or not flat_after_folder:
                raise ValueError("扫描后 dark/flat 目录需要同时提供。")
            calib_after = load_calibration_set(
                dark_folder=dark_after_folder,
                flat_folder=flat_after_folder,
                config=config,
            )
        else:
            # 无时间漂移模式：前后标定集相同
            calib_after = calib_before

    if calib_before.point_ids != calib_after.point_ids:
        raise ValueError(
            f"Flat 点位 ID 不一致: {calib_before.point_ids} vs {calib_after.point_ids}"
        )

    knots_before = build_piecewise_knots(calib_before)
    knots_after = build_piecewise_knots(calib_after)

    metrics = MetricsCollector()
    processed_count = 0
    failed_count = 0
    total = len(projection_paths)

    for idx, proj_path in enumerate(projection_paths):
        alpha = 0.0 if total <= 1 else float(idx / (total - 1))
        file_name = os.path.basename(proj_path)

        try:
            projection = load_image(proj_path, raw_shape=raw_shape)
            dark_t = interpolate_array(calib_before.dark_avg, calib_after.dark_avg, alpha)
            knots_t = interpolate_knots(knots_before, knots_after, alpha)

            projection_dark = projection - dark_t
            corrected = apply_piecewise_linear_correction(
                projection_dark,
                x_knots=knots_t.x_knots,
                y_knots=knots_t.y_knots,
                epsilon=config.epsilon,
            )
            corrected_linear = np.clip(corrected, config.epsilon, None).astype(
                np.float32, copy=False
            )

            stem, _ = os.path.splitext(file_name)
            out_path = os.path.join(output_projection_dir, f"{stem}.tif")
            tiff.imwrite(out_path, corrected_linear, dtype=np.float32)

            air_mean = compute_air_roi_mean(corrected_linear, config.reference_roi)
            stripe_image = corrected_linear
            if config.reference_roi is not None:
                x0, y0, x1, y1 = _normalize_roi(config.reference_roi, corrected_linear.shape)
                stripe_image = corrected_linear[y0:y1, x0:x1]
            stripe_strength = compute_stripe_strength(stripe_image)
            metrics.add(idx + 1, file_name, air_mean, stripe_strength)
            processed_count += 1
        except Exception:
            failed_count += 1

        if progress_callback:
            progress_callback(
                idx + 1,
                total,
                f"处理进度 {idx + 1}/{total}: {file_name}",
            )

    metrics_csv_path = os.path.join(output_folder, "quality_metrics.csv")
    metrics.save_csv(metrics_csv_path)

    summary_json_path = os.path.join(output_folder, "run_summary.json")
    summary_payload = {
        "processed_count": processed_count,
        "failed_count": failed_count,
        "total_count": total,
        "num_points": int(config.num_points),
        "point_ids_before": list(calib_before.point_ids),
        "point_ids_after": list(calib_after.point_ids),
        "metrics_summary": metrics.summary(),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return ProcessingResult(
        processed_count=processed_count,
        failed_count=failed_count,
        output_folder=output_folder,
        metrics_csv_path=metrics_csv_path,
        summary_json_path=summary_json_path,
    )


def run_lifton2019_model_pipeline(
    *,
    projection_folder: str,
    model_folder: str,
    output_folder: str,
    config: ProcessingConfig,
    air_roi: Tuple[int, int, int, int],
    progress_callback: ProgressCallback = None,
) -> ProcessingResult:
    if config.num_points < 2:
        raise ValueError("num_points must be >= 2.")

    if not os.path.isdir(model_folder):
        raise ValueError(f"Model folder does not exist: {model_folder}")

    x_path = os.path.join(model_folder, "calibration_x_knots.npy")
    y_path = os.path.join(model_folder, "calibration_y_refs.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise ValueError("模型目录缺少 calibration_x_knots.npy 或 calibration_y_refs.npy")

    dark_path = os.path.join(model_folder, "calibration_dark_knots.npy")
    flat_path = os.path.join(model_folder, "calibration_flat_knots.npy")

    x_knots = np.load(x_path).astype(np.float32, copy=False)
    y_knots = np.load(y_path).astype(np.float32, copy=False)
    if x_knots.ndim != 3:
        raise ValueError(f"x_knots 维度错误，期望 3 维，实际 {x_knots.ndim}")
    if y_knots.ndim != 1:
        raise ValueError(f"y_knots 维度错误，期望 1 维，实际 {y_knots.ndim}")
    if x_knots.shape[0] != y_knots.shape[0]:
        raise ValueError("x_knots 与 y_knots 点数不一致。")

    dark_knots = None
    if os.path.exists(dark_path):
        dark_knots = np.load(dark_path).astype(np.float32, copy=False)
        if dark_knots.shape != x_knots.shape:
            raise ValueError(
                f"dark_knots 尺寸不匹配: {dark_knots.shape} vs {x_knots.shape}"
            )

    flat_knots = None
    if os.path.exists(flat_path):
        flat_knots = np.load(flat_path).astype(np.float32, copy=False)
        if flat_knots.shape != x_knots.shape:
            raise ValueError(
                f"flat_knots 尺寸不匹配: {flat_knots.shape} vs {x_knots.shape}"
            )

    if progress_callback:
        progress_callback(0, 1, "正在加载投影数据...")

    os.makedirs(output_folder, exist_ok=True)
    output_projection_dir = os.path.join(output_folder, "projections_corrected_from_model")
    os.makedirs(output_projection_dir, exist_ok=True)
    export_interpolated_backgrounds = bool(dark_knots is not None and flat_knots is not None)
    output_bg_root = None
    output_flat_bg_dir = None
    output_dark_bg_dir = None
    if export_interpolated_backgrounds:
        output_bg_root = os.path.join(output_folder, "interpolated_backgrounds")
        output_flat_bg_dir = os.path.join(output_bg_root, "flat")
        output_dark_bg_dir = os.path.join(output_bg_root, "dark")
        os.makedirs(output_flat_bg_dir, exist_ok=True)
        os.makedirs(output_dark_bg_dir, exist_ok=True)

    projection_paths = collect_image_files(projection_folder)
    if not projection_paths:
        raise ValueError(f"No projection files found: {projection_folder}")

    model_shape = (int(x_knots.shape[1]), int(x_knots.shape[2]))
    raw_shape: Tuple[int, int] = (config.raw_height, config.raw_width)
    if model_shape != raw_shape:
        raise ValueError(
            f"模型尺寸 {model_shape} 与当前 RAW 设置 {raw_shape} 不一致。"
        )

    flat_axis_values = None
    flat_col_profiles = None
    flat_stack = None
    dark_stack = None
    if dark_knots is not None and flat_knots is not None:
        flat_stack = flat_knots
        dark_stack = dark_knots
        flat_axis_values = np.asarray(
            [_roi_mean(flat_stack[i], air_roi) for i in range(flat_stack.shape[0])],
            dtype=np.float32,
        )
        flat_col_profiles = np.asarray(
            [_roi_column_profile(flat_stack[i], air_roi) for i in range(flat_stack.shape[0])],
            dtype=np.float32,
        )
        # Keep all model stacks/profile in one deterministic order.
        order = np.argsort(flat_axis_values)
        flat_axis_values = flat_axis_values[order]
        flat_col_profiles = flat_col_profiles[order]
        flat_stack = flat_stack[order]
        dark_stack = dark_stack[order]

    metrics = MetricsCollector()
    processed_count = 0
    failed_count = 0
    total = len(projection_paths)
    export_bg_indices = {0} if total <= 1 else {0, total - 1}
    blend_coeff_records = []
    blend_pair_counts = {}
    denom_floor_ratio = 0.005
    roi_denom_records = []
    denom_floor_records = []
    roi_norm_scale_records = []
    interpolated_background_saved_count = 0

    for idx, proj_path in enumerate(projection_paths):
        file_name = os.path.basename(proj_path)
        try:
            projection = load_image(proj_path, raw_shape=raw_shape).astype(np.float32, copy=False)
            corrected_linear = None
            flat_t_for_export = None
            dark_t_for_export = None

            # Preferred path: dynamic per-frame flat/dark interpolation + ratio correction.
            if (
                dark_knots is not None
                and flat_axis_values is not None
                and flat_stack is not None
                and flat_col_profiles is not None
                and dark_stack is not None
            ):
                projection_profile = _roi_column_profile(projection, air_roi)
                projection_flat_mean = _roi_mean(projection, air_roi)
                i0, i1 = _select_bracketing_pair(flat_axis_values, projection_flat_mean)
                coeff_c = _estimate_profile_blend_coeff(
                    projection_profile,
                    flat_col_profiles[i0],
                    flat_col_profiles[i1],
                    epsilon=config.epsilon,
                )
                blend_coeff_records.append(float(coeff_c))
                pair_key = f"{int(i0)}-{int(i1)}"
                blend_pair_counts[pair_key] = int(blend_pair_counts.get(pair_key, 0) + 1)

                # Single global coeff from flat ROI profile, then blend full-frame flat/dark.
                flat_t = (
                    coeff_c * flat_stack[i0] + (1.0 - coeff_c) * flat_stack[i1]
                ).astype(np.float32, copy=False)
                dark_t = (
                    coeff_c * dark_stack[i0] + (1.0 - coeff_c) * dark_stack[i1]
                ).astype(np.float32, copy=False)
                flat_t_for_export = flat_t
                dark_t_for_export = dark_t
                denom_raw = flat_t - dark_t
                roi_denom = max(float(_roi_mean(denom_raw, air_roi)), config.epsilon)
                denom_floor = max(config.epsilon, denom_floor_ratio * roi_denom)
                denom = np.where(denom_raw < denom_floor, denom_floor, denom_raw)
                roi_denom_records.append(roi_denom)
                denom_floor_records.append(denom_floor)
                corrected_linear = np.clip((projection - dark_t) / denom, config.epsilon, None).astype(
                    np.float32, copy=False
                )
            else:
                # Fallback path: piecewise linear inverse correction on dark-corrected projection.
                work = projection
                if dark_knots is not None and flat_axis_values is not None:
                    projection_air = _roi_mean(projection, air_roi)
                    dark_t = _interpolate_stack_by_scalar(
                        dark_knots, flat_axis_values, projection_air, epsilon=config.epsilon
                    )
                    work = projection - dark_t

                corrected = apply_piecewise_linear_correction(
                    work,
                    x_knots=x_knots,
                    y_knots=y_knots,
                    epsilon=config.epsilon,
                )
                corrected_linear = np.clip(corrected, config.epsilon, None).astype(
                    np.float32, copy=False
                )

            # Enforce per-frame air ROI baseline to suppress sequence-wise brightness drift.
            air_scale = max(_roi_mean(corrected_linear, air_roi), config.epsilon)
            corrected_linear = np.clip(
                corrected_linear / air_scale,
                config.epsilon,
                None,
            ).astype(np.float32, copy=False)
            roi_norm_scale_records.append(float(air_scale))

            stem, _ = os.path.splitext(file_name)
            out_path = os.path.join(output_projection_dir, f"{stem}.tif")
            tiff.imwrite(out_path, corrected_linear, dtype=np.float32)
            if (
                export_interpolated_backgrounds
                and (idx in export_bg_indices)
                and output_flat_bg_dir is not None
                and output_dark_bg_dir is not None
                and flat_t_for_export is not None
                and dark_t_for_export is not None
            ):
                flat_bg_path = os.path.join(output_flat_bg_dir, f"{stem}.tif")
                dark_bg_path = os.path.join(output_dark_bg_dir, f"{stem}.tif")
                tiff.imwrite(flat_bg_path, flat_t_for_export.astype(np.float32, copy=False), dtype=np.float32)
                tiff.imwrite(dark_bg_path, dark_t_for_export.astype(np.float32, copy=False), dtype=np.float32)
                interpolated_background_saved_count += 1

            air_mean = compute_air_roi_mean(corrected_linear, config.reference_roi)
            stripe_image = corrected_linear
            if config.reference_roi is not None:
                x0, y0, x1, y1 = _normalize_roi(config.reference_roi, corrected_linear.shape)
                stripe_image = corrected_linear[y0:y1, x0:x1]
            stripe_strength = compute_stripe_strength(stripe_image)
            metrics.add(idx + 1, file_name, air_mean, stripe_strength)
            processed_count += 1
        except Exception:
            failed_count += 1

        if progress_callback:
            progress_callback(idx + 1, total, f"处理进度 {idx + 1}/{total}: {file_name}")

    metrics_csv_path = os.path.join(output_folder, "quality_metrics_model.csv")
    metrics.save_csv(metrics_csv_path)

    summary_json_path = os.path.join(output_folder, "run_summary_model.json")
    dynamic_ratio_enabled = bool(
        dark_knots is not None
        and flat_axis_values is not None
        and flat_stack is not None
        and flat_col_profiles is not None
        and dark_stack is not None
    )
    summary_payload = {
        "processed_count": processed_count,
        "failed_count": failed_count,
        "total_count": total,
        "num_points": int(x_knots.shape[0]),
        "model_folder": model_folder,
        "air_roi": [int(v) for v in air_roi],
        "correction_mode": (
            "dynamic_flat_dark_ratio"
            if dynamic_ratio_enabled
            else "piecewise_inverse_fallback"
        ),
        "axis_match_method": (
            "flat_roi_bracket_then_profile_coeff"
            if dynamic_ratio_enabled
            else "roi_mean"
        ),
        "denominator_floor_ratio": denom_floor_ratio if dynamic_ratio_enabled else None,
        "air_roi_post_normalization": True,
        "export_interpolated_backgrounds": export_interpolated_backgrounds,
        "interpolated_background_saved_count": int(interpolated_background_saved_count),
        "interpolated_background_flat_dir": output_flat_bg_dir,
        "interpolated_background_dark_dir": output_dark_bg_dir,
        "metrics_summary": metrics.summary(),
    }
    if blend_coeff_records:
        c = np.asarray(blend_coeff_records, dtype=np.float64)
        summary_payload["blend_coeff_stats"] = {
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
            "min": float(np.min(c)),
            "max": float(np.max(c)),
            "count": int(c.size),
        }
        summary_payload["blend_pair_counts"] = dict(blend_pair_counts)
    if roi_denom_records:
        d = np.asarray(roi_denom_records, dtype=np.float64)
        f = np.asarray(denom_floor_records, dtype=np.float64)
        summary_payload["denominator_stats"] = {
            "roi_denom_mean": float(np.mean(d)),
            "roi_denom_std": float(np.std(d)),
            "roi_denom_min": float(np.min(d)),
            "roi_denom_max": float(np.max(d)),
            "floor_mean": float(np.mean(f)),
            "floor_std": float(np.std(f)),
            "floor_min": float(np.min(f)),
            "floor_max": float(np.max(f)),
            "count": int(d.size),
        }
    if roi_norm_scale_records:
        s = np.asarray(roi_norm_scale_records, dtype=np.float64)
        summary_payload["air_roi_pre_normalization_stats"] = {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "count": int(s.size),
        }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return ProcessingResult(
        processed_count=processed_count,
        failed_count=failed_count,
        output_folder=output_folder,
        metrics_csv_path=metrics_csv_path,
        summary_json_path=summary_json_path,
    )
