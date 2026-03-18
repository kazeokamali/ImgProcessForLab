from typing import Tuple

import numpy as np

from .models import ReconstructionConfig, ReconstructionDerived


def _is_iterative_algorithm(algorithm: str) -> bool:
    name = (algorithm or "").strip().lower()
    return ("sirt" in name) or ("cgls" in name)


def validate_config(config: ReconstructionConfig):
    if config.projection_count < 1:
        raise ValueError("投影数量必须 >= 1。")
    if config.sod_mm <= 0 or config.sdd_mm <= 0:
        raise ValueError("SOD 和 SDD 必须为正数。")
    if config.sdd_mm <= config.sod_mm:
        raise ValueError("SDD 必须大于 SOD。")
    if config.detector_pixel_size_x_mm <= 0 or config.detector_pixel_size_y_mm <= 0:
        raise ValueError("探测器像素尺寸必须为正数。")
    if abs(config.angle_step_deg) < 1e-12:
        raise ValueError("每帧角度间隔不能为 0。")
    if config.recon_nx < 1 or config.recon_ny < 1 or config.recon_nz < 1:
        raise ValueError("重构体尺寸必须 >= 1。")
    if config.iterative_iterations < 0:
        raise ValueError("迭代重构次数必须 >= 0。")
    if _is_iterative_algorithm(config.algorithm) and config.iterative_iterations <= 0:
        raise ValueError("当前算法为迭代重构，请将迭代重构次数设为 >= 1。")
    if config.refine_iterations < 0:
        raise ValueError("后处理扩散次数必须 >= 0。")
    if config.refine_step < 0:
        raise ValueError("后处理扩散步长必须 >= 0。")


def compute_derived(config: ReconstructionConfig) -> ReconstructionDerived:
    validate_config(config)
    magnification = float(config.sdd_mm / config.sod_mm)
    voxel_size_x_mm = float(config.detector_pixel_size_x_mm / magnification)
    voxel_size_y_mm = float(config.detector_pixel_size_y_mm / magnification)
    voxel_size_z_mm = voxel_size_y_mm
    total_scan_angle_deg = float((config.projection_count - 1) * config.angle_step_deg)
    return ReconstructionDerived(
        magnification=magnification,
        voxel_size_x_mm=voxel_size_x_mm,
        voxel_size_y_mm=voxel_size_y_mm,
        voxel_size_z_mm=voxel_size_z_mm,
        total_scan_angle_deg=total_scan_angle_deg,
    )


def generate_angles(config: ReconstructionConfig) -> np.ndarray:
    validate_config(config)
    return (
        config.start_angle_deg
        + np.arange(config.projection_count, dtype=np.float64) * config.angle_step_deg
    ).astype(np.float32, copy=False)


def build_stage1_plan(config: ReconstructionConfig) -> Tuple[ReconstructionDerived, np.ndarray]:
    derived = compute_derived(config)
    angles = generate_angles(config)
    return derived, angles
