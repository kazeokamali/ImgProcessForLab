import json
import os
from typing import Callable, Optional, Tuple

import numpy as np
import tifffile as tiff

from .calibration_builder import apply_piecewise_linear_correction, build_piecewise_knots
from .drift_interpolator import interpolate_array, interpolate_knots
from .io_loader import collect_image_files, load_calibration_set, load_image
from .metrics import MetricsCollector, compute_air_roi_mean, compute_stripe_strength
from .models import ProcessingConfig, ProcessingResult


ProgressCallback = Optional[Callable[[int, int, str], None]]


def run_lifton2019_pipeline(
    *,
    projection_folder: str,
    dark_before_folder: str,
    flat_before_folder: str,
    dark_after_folder: str,
    flat_after_folder: str,
    output_folder: str,
    config: ProcessingConfig,
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

    if progress_callback:
        progress_callback(0, len(projection_paths), "正在加载标定数据...")

    calib_before = load_calibration_set(
        dark_folder=dark_before_folder,
        flat_folder=flat_before_folder,
        config=config,
    )
    calib_after = load_calibration_set(
        dark_folder=dark_after_folder,
        flat_folder=flat_after_folder,
        config=config,
    )

    if calib_before.point_ids != calib_after.point_ids:
        raise ValueError(
            f"Flat point IDs mismatch between before/after: {calib_before.point_ids} vs {calib_after.point_ids}"
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
            corrected = np.clip(corrected, config.epsilon, None)
            neg_log = -np.log(corrected).astype(np.float32, copy=False)

            stem, _ = os.path.splitext(file_name)
            out_path = os.path.join(output_projection_dir, f"{stem}.tif")
            tiff.imwrite(out_path, neg_log, dtype=np.float32)

            air_mean = compute_air_roi_mean(neg_log, config.reference_roi)
            stripe_strength = compute_stripe_strength(neg_log)
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
