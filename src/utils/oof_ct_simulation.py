import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
from scipy import ndimage


ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class OOFSimulationConfig:
    sim_z: int
    detector_width: int
    num_angles: int
    phantom_type: str  # "cylinder" | "sphere"
    base_radius_px: float
    base_intensity: float

    add_oof_material: bool
    oof_type: str  # "ring_shell" | "side_blobs"
    oof_intensity: float
    oof_size_px: float

    auto_xy_margin_px: int = 48


def _emit(progress_callback: ProgressCallback, done: int, total: int, message: str):
    if progress_callback is not None:
        progress_callback(int(done), int(total), message)


def _center_crop_xy(volume: np.ndarray, target_w: int) -> np.ndarray:
    nz, ny, nx = volume.shape
    target_w = int(min(target_w, nx, ny))
    x0 = (nx - target_w) // 2
    y0 = (ny - target_w) // 2
    return volume[:, y0 : y0 + target_w, x0 : x0 + target_w]


def _disc_mask_2d(height: int, width: int, diameter: float) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return rr <= (float(diameter) * 0.5)


def _center_crop_disc_mask_xy(volume: np.ndarray, target_w: int) -> np.ndarray:
    local = _center_crop_xy(volume, target_w)
    mask = _disc_mask_2d(local.shape[1], local.shape[2], diameter=float(target_w))
    out = local.copy()
    out[:, ~mask] = 0.0
    return out


def _auto_sim_xy(config: OOFSimulationConfig) -> int:
    det_w = int(config.detector_width)
    margin = int(max(8, config.auto_xy_margin_px))
    by_detector = det_w + 2 * margin
    by_radius = int(np.ceil(2.0 * float(config.base_radius_px) + 2.0 * margin))
    sim_xy = max(by_detector, by_radius)
    if sim_xy % 2 != 0:
        sim_xy += 1
    return int(sim_xy)


def _make_base_volume(sim_xy: int, sim_z: int, phantom_type: str, radius_px: float, intensity: float) -> np.ndarray:
    sim_xy = int(sim_xy)
    sim_z = int(sim_z)
    radius_px = float(max(2.0, radius_px))
    intensity = float(intensity)

    yy, xx = np.meshgrid(
        np.arange(sim_xy, dtype=np.float32),
        np.arange(sim_xy, dtype=np.float32),
        indexing="ij",
    )
    cx = (sim_xy - 1) * 0.5
    cy = (sim_xy - 1) * 0.5
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    volume = np.zeros((sim_z, sim_xy, sim_xy), dtype=np.float32)
    if phantom_type == "sphere":
        cz = (sim_z - 1) * 0.5
        for z in range(sim_z):
            dz = float(z - cz)
            r_xy_sq = radius_px * radius_px - dz * dz
            if r_xy_sq <= 0:
                continue
            mask = r2 <= r_xy_sq
            volume[z][mask] = intensity
    else:
        mask = r2 <= radius_px * radius_px
        volume[:, mask] = intensity
    return volume


def _add_oof_material(
    volume: np.ndarray,
    detector_width: int,
    oof_type: str,
    oof_intensity: float,
    oof_size_px: float,
) -> np.ndarray:
    out = volume.copy()
    nz, ny, nx = out.shape
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    half_local = float(detector_width) * 0.5
    size_px = float(max(2.0, oof_size_px))

    yy, xx = np.meshgrid(
        np.arange(ny, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        indexing="ij",
    )
    radius_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outside_local = radius_map > half_local

    if oof_type == "side_blobs":
        blob_radius = size_px
        left_cx = cx - half_local - blob_radius * 0.9
        right_cx = cx + half_local + blob_radius * 0.9
        cy0 = cy
        blob_left = ((xx - left_cx) ** 2 + (yy - cy0) ** 2) <= blob_radius * blob_radius
        blob_right = ((xx - right_cx) ** 2 + (yy - cy0) ** 2) <= blob_radius * blob_radius
        mask = outside_local & (blob_left | blob_right)
    else:
        r_inner = half_local + max(2.0, size_px * 0.2)
        r_outer = min(nx, ny) * 0.48
        mask = outside_local & (radius_map >= r_inner) & (radius_map <= r_outer)

    if np.any(mask):
        out[:, mask] = float(oof_intensity)
    return out


def _forward_project_parallel(
    volume: np.ndarray,
    num_angles: int,
    detector_width: int,
    progress_callback: ProgressCallback = None,
    stage_prefix: str = "",
) -> np.ndarray:
    num_angles = int(max(1, num_angles))
    detector_width = int(detector_width)
    nz, ny, nx = volume.shape
    if detector_width > nx:
        raise ValueError("detector_width 不能大于仿真体宽度。")

    x0 = (nx - detector_width) // 2
    x1 = x0 + detector_width
    angles = np.linspace(0.0, 360.0, num_angles, endpoint=False, dtype=np.float32)
    projections = np.zeros((num_angles, nz, detector_width), dtype=np.float32)

    for i, angle in enumerate(angles.tolist()):
        rotated = ndimage.rotate(
            volume,
            angle=float(angle),
            axes=(2, 1),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        proj_full = rotated.sum(axis=1, dtype=np.float32)  # [z, x]
        projections[i] = proj_full[:, x0:x1]
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == num_angles:
            _emit(progress_callback, i + 1, num_angles, f"{stage_prefix}投影模拟 {i + 1}/{num_angles}")
    return projections


def _save_projection_stack(projections: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(projections.shape[0]):
        path = os.path.join(out_dir, f"proj_{i:05d}.tif")
        tiff.imwrite(path, projections[i].astype(np.float32, copy=False), dtype=np.float32)


def _save_slice_stack(volume: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for z in range(volume.shape[0]):
        path = os.path.join(out_dir, f"slice_{z:05d}.tif")
        tiff.imwrite(path, volume[z].astype(np.float32, copy=False), dtype=np.float32)


def _load_slice_volume_from_folder(folder: str) -> np.ndarray:
    files = _collect_tiff_files(folder)
    if not files:
        raise ValueError(f"切片目录为空或不存在: {folder}")

    slices: List[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int]] = None
    for path in files:
        img = tiff.imread(path).astype(np.float32, copy=False)
        if img.ndim == 3:
            img = img[:, :, 0]
        if img.ndim != 2:
            raise ValueError(f"仅支持2D切片，文件异常: {path}")
        if expected_shape is None:
            expected_shape = img.shape
        elif img.shape != expected_shape:
            raise ValueError(
                f"切片尺寸不一致，期望 {expected_shape}，实际 {img.shape}，文件: {path}"
            )
        slices.append(img)

    return np.stack(slices, axis=0).astype(np.float32, copy=False)


def simulate_oof_dataset(
    config: OOFSimulationConfig,
    output_root: str,
    progress_callback: ProgressCallback = None,
) -> Dict[str, str]:
    sim_z = int(config.sim_z)
    detector_width = int(config.detector_width)
    if sim_z <= 0 or detector_width <= 0:
        raise ValueError("仿真尺寸参数必须为正数。")

    sim_xy = _auto_sim_xy(config)
    if detector_width >= sim_xy:
        raise ValueError("自动计算的仿真宽度异常，请检查参数。")

    os.makedirs(output_root, exist_ok=True)
    case_base_root = os.path.join(output_root, "case_base")
    case_oof_root = os.path.join(output_root, "case_oof")
    os.makedirs(case_base_root, exist_ok=True)
    os.makedirs(case_oof_root, exist_ok=True)

    _emit(progress_callback, 0, 100, "生成基础体模...")
    base_volume = _make_base_volume(
        sim_xy=sim_xy,
        sim_z=sim_z,
        phantom_type=config.phantom_type,
        radius_px=float(config.base_radius_px),
        intensity=float(config.base_intensity),
    )
    base_local = _center_crop_disc_mask_xy(base_volume, detector_width)

    _emit(progress_callback, 10, 100, "模拟基础体模投影...")
    base_proj = _forward_project_parallel(
        volume=base_volume,
        num_angles=int(config.num_angles),
        detector_width=detector_width,
        progress_callback=lambda d, t, m: _emit(
            progress_callback,
            10 + int(round((d / max(1, t)) * 30.0)),
            100,
            f"[Base] {m}",
        ),
        stage_prefix="[Base] ",
    )

    _emit(progress_callback, 45, 100, "保存基础数据...")
    base_proj_dir = os.path.join(case_base_root, "projections")
    base_gt_local_dir = os.path.join(case_base_root, "ground_truth_local_slices")
    base_gt_full_dir = os.path.join(case_base_root, "ground_truth_full_slices")
    _save_projection_stack(base_proj, base_proj_dir)
    _save_slice_stack(base_local, base_gt_local_dir)
    _save_slice_stack(base_volume, base_gt_full_dir)

    if config.add_oof_material:
        _emit(progress_callback, 55, 100, "添加超视野材料...")
        oof_volume = _add_oof_material(
            volume=base_volume,
            detector_width=detector_width,
            oof_type=config.oof_type,
            oof_intensity=float(config.oof_intensity),
            oof_size_px=float(config.oof_size_px),
        )
    else:
        oof_volume = base_volume.copy()
    oof_local = _center_crop_disc_mask_xy(oof_volume, detector_width)

    _emit(progress_callback, 60, 100, "模拟超视野场景投影...")
    oof_proj = _forward_project_parallel(
        volume=oof_volume,
        num_angles=int(config.num_angles),
        detector_width=detector_width,
        progress_callback=lambda d, t, m: _emit(
            progress_callback,
            60 + int(round((d / max(1, t)) * 30.0)),
            100,
            f"[OOF] {m}",
        ),
        stage_prefix="[OOF] ",
    )

    _emit(progress_callback, 92, 100, "保存超视野数据...")
    oof_proj_dir = os.path.join(case_oof_root, "projections")
    oof_gt_local_dir = os.path.join(case_oof_root, "ground_truth_local_slices")
    oof_gt_full_dir = os.path.join(case_oof_root, "ground_truth_full_slices")
    _save_projection_stack(oof_proj, oof_proj_dir)
    _save_slice_stack(oof_local, oof_gt_local_dir)
    _save_slice_stack(oof_volume, oof_gt_full_dir)

    metadata = {
        "sim_config": {
            "sim_xy_auto": int(sim_xy),
            "sim_z": int(sim_z),
            "detector_width": int(detector_width),
            "local_ground_truth_geometry": "disc_in_square",
            "num_angles": int(config.num_angles),
            "angle_step_deg": float(360.0 / max(1, int(config.num_angles))),
            "phantom_type": config.phantom_type,
            "base_radius_px": float(config.base_radius_px),
            "base_intensity": float(config.base_intensity),
            "add_oof_material": bool(config.add_oof_material),
            "oof_type": config.oof_type,
            "oof_intensity": float(config.oof_intensity),
            "oof_size_px": float(config.oof_size_px),
            "auto_xy_margin_px": int(config.auto_xy_margin_px),
        },
        "folders": {
            "base_projection_dir": base_proj_dir,
            "base_ground_truth_local_dir": base_gt_local_dir,
            "base_ground_truth_full_dir": base_gt_full_dir,
            "oof_projection_dir": oof_proj_dir,
            "oof_ground_truth_local_dir": oof_gt_local_dir,
            "oof_ground_truth_full_dir": oof_gt_full_dir,
        },
    }
    metadata_path = os.path.join(output_root, "simulation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _emit(progress_callback, 100, 100, "仿真数据生成完成。")
    return {
        "sim_xy_auto": str(sim_xy),
        "base_projection_dir": base_proj_dir,
        "base_ground_truth_local_dir": base_gt_local_dir,
        "base_ground_truth_full_dir": base_gt_full_dir,
        "oof_projection_dir": oof_proj_dir,
        "oof_ground_truth_local_dir": oof_gt_local_dir,
        "oof_ground_truth_full_dir": oof_gt_full_dir,
        "metadata_path": metadata_path,
    }


def simulate_oof_dataset_from_slice_folders(
    base_full_slice_dir: str,
    oof_full_slice_dir: Optional[str],
    detector_width: int,
    num_angles: int,
    output_root: str,
    progress_callback: ProgressCallback = None,
) -> Dict[str, str]:
    if not base_full_slice_dir:
        raise ValueError("base 原始切片目录不能为空。")
    if not os.path.isdir(base_full_slice_dir):
        raise ValueError(f"base 原始切片目录不存在: {base_full_slice_dir}")
    if oof_full_slice_dir and (not os.path.isdir(oof_full_slice_dir)):
        raise ValueError(f"oof 原始切片目录不存在: {oof_full_slice_dir}")

    detector_width = int(detector_width)
    num_angles = int(max(1, num_angles))
    if detector_width <= 0:
        raise ValueError("探测器列数必须为正数。")

    os.makedirs(output_root, exist_ok=True)
    case_base_root = os.path.join(output_root, "case_base")
    case_oof_root = os.path.join(output_root, "case_oof")
    os.makedirs(case_base_root, exist_ok=True)
    os.makedirs(case_oof_root, exist_ok=True)

    _emit(progress_callback, 0, 100, "加载 base 原始切片...")
    base_volume = _load_slice_volume_from_folder(base_full_slice_dir)

    if oof_full_slice_dir:
        _emit(progress_callback, 5, 100, "加载 OOF 原始切片...")
        oof_volume = _load_slice_volume_from_folder(oof_full_slice_dir)
    else:
        _emit(progress_callback, 5, 100, "未提供 OOF 原始切片，默认使用 base 切片。")
        oof_volume = base_volume.copy()
        oof_full_slice_dir = base_full_slice_dir

    if base_volume.shape != oof_volume.shape:
        raise ValueError(
            f"base/oof 切片体尺寸不一致: base={base_volume.shape}, oof={oof_volume.shape}"
        )

    sim_z, ny, nx = [int(v) for v in base_volume.shape]
    if detector_width > min(ny, nx):
        raise ValueError(
            f"探测器列数 {detector_width} 大于输入切片宽高最小值 {min(ny, nx)}。"
        )

    base_local = _center_crop_disc_mask_xy(base_volume, detector_width)
    oof_local = _center_crop_disc_mask_xy(oof_volume, detector_width)

    _emit(progress_callback, 10, 100, "模拟 base 投影...")
    base_proj = _forward_project_parallel(
        volume=base_volume,
        num_angles=num_angles,
        detector_width=detector_width,
        progress_callback=lambda d, t, m: _emit(
            progress_callback,
            10 + int(round((d / max(1, t)) * 35.0)),
            100,
            f"[Base] {m}",
        ),
        stage_prefix="[Base] ",
    )

    _emit(progress_callback, 47, 100, "保存 base 数据...")
    base_proj_dir = os.path.join(case_base_root, "projections")
    base_gt_local_dir = os.path.join(case_base_root, "ground_truth_local_slices")
    _save_projection_stack(base_proj, base_proj_dir)
    _save_slice_stack(base_local, base_gt_local_dir)

    _emit(progress_callback, 55, 100, "模拟 OOF 投影...")
    oof_proj = _forward_project_parallel(
        volume=oof_volume,
        num_angles=num_angles,
        detector_width=detector_width,
        progress_callback=lambda d, t, m: _emit(
            progress_callback,
            55 + int(round((d / max(1, t)) * 35.0)),
            100,
            f"[OOF] {m}",
        ),
        stage_prefix="[OOF] ",
    )

    _emit(progress_callback, 92, 100, "保存 OOF 数据...")
    oof_proj_dir = os.path.join(case_oof_root, "projections")
    oof_gt_local_dir = os.path.join(case_oof_root, "ground_truth_local_slices")
    _save_projection_stack(oof_proj, oof_proj_dir)
    _save_slice_stack(oof_local, oof_gt_local_dir)

    metadata = {
        "sim_config": {
            "source_mode": "user_slice_folders",
            "input_volume_shape_zyx": [sim_z, ny, nx],
            "sim_xy_input": [ny, nx],
            "detector_width": detector_width,
            "local_ground_truth_geometry": "disc_in_square",
            "num_angles": num_angles,
            "angle_step_deg": float(360.0 / max(1, num_angles)),
        },
        "source_folders": {
            "base_full_slice_dir": base_full_slice_dir,
            "oof_full_slice_dir": oof_full_slice_dir,
        },
        "folders": {
            "base_projection_dir": base_proj_dir,
            "base_ground_truth_local_dir": base_gt_local_dir,
            "base_ground_truth_full_dir": base_full_slice_dir,
            "oof_projection_dir": oof_proj_dir,
            "oof_ground_truth_local_dir": oof_gt_local_dir,
            "oof_ground_truth_full_dir": oof_full_slice_dir,
        },
    }
    metadata_path = os.path.join(output_root, "simulation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _emit(progress_callback, 100, 100, "用户切片仿真数据生成完成。")
    return {
        "sim_xy_auto": str(min(ny, nx)),
        "base_projection_dir": base_proj_dir,
        "base_ground_truth_local_dir": base_gt_local_dir,
        "base_ground_truth_full_dir": base_full_slice_dir,
        "oof_projection_dir": oof_proj_dir,
        "oof_ground_truth_local_dir": oof_gt_local_dir,
        "oof_ground_truth_full_dir": oof_full_slice_dir,
        "metadata_path": metadata_path,
        "source_mode": "user_slice_folders",
    }


def _collect_tiff_files(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".tif", ".tiff"}:
            files.append(os.path.join(folder, name))
    return files


def _crop_roi(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return image
    h_img, w_img = image.shape
    x0 = max(0, min(w_img - 1, x))
    y0 = max(0, min(h_img - 1, y))
    x1 = max(x0 + 1, min(w_img, x0 + w))
    y1 = max(y0 + 1, min(h_img, y0 + h))
    return image[y0:y1, x0:x1]


def _align_shapes(gt: np.ndarray, recon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(gt.shape[0], recon.shape[0])
    w = min(gt.shape[1], recon.shape[1])

    def _center_crop(img: np.ndarray, hh: int, ww: int) -> np.ndarray:
        y0 = (img.shape[0] - hh) // 2
        x0 = (img.shape[1] - ww) // 2
        return img[y0 : y0 + hh, x0 : x0 + ww]

    return _center_crop(gt, h, w), _center_crop(recon, h, w)


def compare_recon_to_ground_truth(
    ground_truth_dir: str,
    recon_dir: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    slice_step: int = 1,
) -> Dict[str, float]:
    gt_files = _collect_tiff_files(ground_truth_dir)
    recon_files = _collect_tiff_files(recon_dir)
    if not gt_files or not recon_files:
        raise ValueError("ground truth 或 reconstruction 目录为空。")

    n = min(len(gt_files), len(recon_files))
    step = max(1, int(slice_step))
    indices = list(range(0, n, step))
    if not indices:
        raise ValueError("有效切片数量为 0。")

    mae_list: List[float] = []
    rmse_list: List[float] = []
    psnr_list: List[float] = []
    ncc_list: List[float] = []

    for i in indices:
        gt = tiff.imread(gt_files[i]).astype(np.float32, copy=False)
        recon = tiff.imread(recon_files[i]).astype(np.float32, copy=False)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        if recon.ndim == 3:
            recon = recon[:, :, 0]

        gt, recon = _align_shapes(gt, recon)
        gt = _crop_roi(gt, roi)
        recon = _crop_roi(recon, roi)

        diff = recon - gt
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        gt_range = float(np.max(gt) - np.min(gt))
        psnr = float(20.0 * np.log10((gt_range + 1e-6) / (rmse + 1e-6)))

        gt_f = gt.reshape(-1)
        rc_f = recon.reshape(-1)
        gt_c = gt_f - float(np.mean(gt_f))
        rc_c = rc_f - float(np.mean(rc_f))
        denom = float(np.sqrt(np.sum(gt_c * gt_c) * np.sum(rc_c * rc_c)) + 1e-12)
        ncc = float(np.sum(gt_c * rc_c) / denom)

        mae_list.append(mae)
        rmse_list.append(rmse)
        psnr_list.append(psnr)
        ncc_list.append(ncc)

    return {
        "slice_count_used": float(len(indices)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "ncc_mean": float(np.mean(ncc_list)),
        "ncc_std": float(np.std(ncc_list)),
    }
