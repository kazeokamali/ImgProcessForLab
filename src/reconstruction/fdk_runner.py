import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import tifffile as tiff

from .models import ReconstructionConfig
from .pipeline import build_stage1_plan, validate_config


ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class ReconstructionRunResult:
    output_slice_dir: str
    summary_json_path: str
    angles_csv_path: str
    stage1_plan_json_path: str
    slice_count: int
    projection_count: int
    elapsed_seconds: float
    backend_used: str
    backend_note: str


@dataclass
class ReconstructionPreviewResult:
    preview_path: str
    preview_dir: str
    z_index: int
    elapsed_seconds: float
    backend_used: str
    backend_note: str


def _emit_progress(progress_callback: ProgressCallback, done: int, total: int, message: str):
    if progress_callback is not None:
        progress_callback(int(done), int(total), message)


def _estimate_astra_required_bytes(
    det_h: int,
    det_w: int,
    n_proj: int,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    # ASTRA 3D path at least needs full projection stack + reconstruction volume.
    projection_bytes = int(det_h) * int(det_w) * int(n_proj) * 4
    volume_bytes = int(nx) * int(ny) * int(nz) * 4
    # Conservative overhead factor for temporary buffers/intermediates.
    # Can be tuned by env RECON_ASTRA_EST_FACTOR (default 2.2).
    try:
        factor = float(os.environ.get("RECON_ASTRA_EST_FACTOR", "2.2"))
    except Exception:
        factor = 2.2
    factor = max(1.2, min(factor, 8.0))
    return int((projection_bytes + volume_bytes) * factor)


def _get_cuda_free_bytes() -> Optional[int]:
    try:
        import cupy as _cp  # type: ignore

        free_bytes, _ = _cp.cuda.runtime.memGetInfo()
        return int(free_bytes)
    except Exception:
        return None


def _load_projection(path: str) -> np.ndarray:
    if path.lower().endswith(".raw"):
        raise ValueError("Stage-3 does not support RAW direct reconstruction; convert to tif/tiff first.")
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.ndim != 2:
        raise ValueError(f"Invalid projection image dimensions: {path}, ndim={img.ndim}")
    return np.asarray(img, dtype=np.float32)


def _bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    x0c = np.clip(x0, 0, w - 2)
    x1c = np.clip(x1, 1, w - 1)
    y0c = np.clip(y0, 0, h - 2)
    y1c = np.clip(y1, 1, h - 1)

    fx = (x - x0c).astype(np.float32, copy=False)
    fy = (y - y0c).astype(np.float32, copy=False)

    ia = image[y0c, x0c]
    ib = image[y0c, x1c]
    ic = image[y1c, x0c]
    id_ = image[y1c, x1c]

    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    out = wa * ia + wb * ib + wc * ic + wd * id_
    out = out.astype(np.float32, copy=False)
    out[~valid] = 0.0
    return out


def _filter_projection_fdk(
    projection: np.ndarray,
    sdd_mm: float,
    pixel_x_mm: float,
    pixel_y_mm: float,
    filter_name: str,
) -> np.ndarray:
    h, w = projection.shape
    u = (np.arange(w, dtype=np.float32) - (w - 1) * 0.5) * float(pixel_x_mm)
    v = (np.arange(h, dtype=np.float32) - (h - 1) * 0.5) * float(pixel_y_mm)

    weight = float(sdd_mm) / np.sqrt(float(sdd_mm) ** 2 + u[None, :] ** 2 + v[:, None] ** 2)
    p = projection.astype(np.float32, copy=False) * weight.astype(np.float32, copy=False)

    freq = np.fft.rfftfreq(w, d=float(pixel_x_mm)).astype(np.float32, copy=False)
    ramp = np.abs(freq)
    if filter_name.lower().startswith("shepp"):
        x = np.pi * freq / (freq[-1] + 1e-12)
        ramp = ramp * np.sinc(x / np.pi)
    elif filter_name.lower().startswith("hann"):
        ramp = ramp * (0.5 + 0.5 * np.cos(np.pi * freq / (freq[-1] + 1e-12)))
    elif filter_name.lower().startswith("hamming"):
        ramp = ramp * (0.54 + 0.46 * np.cos(np.pi * freq / (freq[-1] + 1e-12)))

    pf = np.fft.rfft(p, axis=1)
    pf *= ramp[None, :]
    filtered = np.fft.irfft(pf, n=w, axis=1)
    return filtered.astype(np.float32, copy=False)


def _refine_slice_diffusion(slice_img: np.ndarray, iterations: int, step: float) -> np.ndarray:
    if iterations <= 0:
        return slice_img
    out = slice_img.astype(np.float32, copy=False)
    step = float(np.clip(step, 0.0, 0.24))
    for _ in range(iterations):
        lap = (
            np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            - 4.0 * out
        )
        out = out + step * lap
        out = np.clip(out, 0.0, None)
    return out.astype(np.float32, copy=False)


def _try_import_astra() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import astra  # type: ignore
    except Exception as e:
        return None, str(e)
    return astra, None


def _astra_filter_name(filter_name: str) -> str:
    name = (filter_name or "").strip().lower()
    if name.startswith("ram"):
        return "ram-lak"
    if name.startswith("shepp"):
        return "shepp-logan"
    if name.startswith("hann"):
        return "hann"
    if name.startswith("hamming"):
        return "hamming"
    return "ram-lak"


def _algorithm_key(name: str) -> str:
    text = (name or "").strip().lower()
    if ("fdk" in text) and ("cgls" in text):
        return "fdk_cgls"
    if "sirt" in text:
        return "sirt"
    if "cgls" in text:
        return "cgls"
    return "fdk"


def _is_iterative_algorithm(name: str) -> bool:
    key = _algorithm_key(name)
    return key in {"sirt", "cgls", "fdk_cgls"}


def _get_astra_option_base() -> Dict[str, Any]:
    option: Dict[str, Any] = {}
    gpu_idx_str = os.environ.get("ASTRA_GPU_INDEX", "").strip()
    if gpu_idx_str:
        try:
            option["GPUindex"] = int(gpu_idx_str)
        except ValueError:
            pass
    return option


def _build_astra_proj_geom(
    astra_module: Any,
    config: ReconstructionConfig,
    det_h: int,
    det_w: int,
    angles_rad: np.ndarray,
):
    origin_det = float(config.sdd_mm) - float(config.sod_mm)
    if origin_det <= 0:
        raise ValueError("Invalid ASTRA geometry: require SDD > SOD.")

    proj_geom = astra_module.create_proj_geom(
        "cone",
        float(config.detector_pixel_size_x_mm),
        float(config.detector_pixel_size_y_mm),
        int(det_h),
        int(det_w),
        angles_rad.astype(np.float32, copy=False),
        float(config.sod_mm),
        origin_det,
    )

    cor_offset = float(config.cor_offset_px)
    if abs(cor_offset) > 1e-12:
        proj_geom_vec = astra_module.geom_2vec(proj_geom)
        vectors = np.asarray(proj_geom_vec["Vectors"], dtype=np.float32).copy()
        vectors[:, 3:6] -= np.float32(cor_offset) * vectors[:, 6:9]
        proj_geom_vec["Vectors"] = vectors
        return proj_geom_vec

    return proj_geom


def _build_astra_volume_geom(
    astra_module: Any,
    nx: int,
    ny: int,
    nz: int,
    voxel_x_mm: float,
    voxel_y_mm: float,
    voxel_z_mm: float,
):
    min_x = -0.5 * float(nx) * float(voxel_x_mm)
    max_x = +0.5 * float(nx) * float(voxel_x_mm)
    min_y = -0.5 * float(ny) * float(voxel_y_mm)
    max_y = +0.5 * float(ny) * float(voxel_y_mm)
    min_z = -0.5 * float(nz) * float(voxel_z_mm)
    max_z = +0.5 * float(nz) * float(voxel_z_mm)
    return astra_module.create_vol_geom(
        int(ny),
        int(nx),
        int(nz),
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
    )


def _load_projections_stack(
    projection_files: List[str],
    det_h: int,
    det_w: int,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    n_proj = len(projection_files)
    projections = np.empty((det_h, n_proj, det_w), dtype=np.float32)
    _emit_progress(progress_callback, 2, 100, "ASTRA: loading projection stack...")
    for i, path in enumerate(projection_files):
        if stop_requested and stop_requested():
            raise RuntimeError("Reconstruction cancelled by user.")
        proj = _load_projection(path)
        if proj.shape != (det_h, det_w):
            raise ValueError(
                f"Projection shape mismatch: {os.path.basename(path)} -> {proj.shape}, expected {(det_h, det_w)}"
            )
        projections[:, i, :] = proj
        if i == 0 or (i + 1) % 20 == 0 or (i + 1) == n_proj:
            p = 2 + int(round((i + 1) / max(1, n_proj) * 33.0))
            _emit_progress(progress_callback, p, 100, f"ASTRA: loaded projection {i + 1}/{n_proj}")
    return projections


def _apply_post_refine_if_needed(
    volume: np.ndarray,
    config: ReconstructionConfig,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
):
    if int(config.refine_iterations) <= 0:
        return
    _emit_progress(progress_callback, 85, 100, "ASTRA/CPU: applying post-refine diffusion...")
    for zi in range(volume.shape[0]):
        if stop_requested and stop_requested():
            raise RuntimeError("Reconstruction cancelled by user.")
        volume[zi] = _refine_slice_diffusion(
            volume[zi],
            iterations=int(config.refine_iterations),
            step=float(config.refine_step),
        )


def _write_volume_slices(
    volume: np.ndarray,
    output_slice_dir: str,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
):
    _emit_progress(progress_callback, 92, 100, "Writing output slices...")
    for zi in range(volume.shape[0]):
        if stop_requested and stop_requested():
            raise RuntimeError("Reconstruction cancelled by user.")
        out_path = os.path.join(output_slice_dir, f"slice_{zi:05d}.tif")
        tiff.imwrite(out_path, volume[zi].astype(np.float32, copy=False), dtype=np.float32)
        if zi == 0 or (zi + 1) % 50 == 0 or (zi + 1) == volume.shape[0]:
            p = 92 + int(round((zi + 1) / max(1, volume.shape[0]) * 7.0))
            _emit_progress(progress_callback, p, 100, f"Wrote slice {zi + 1}/{volume.shape[0]}")


def _run_astra_reconstruction(
    astra_module: Any,
    config: ReconstructionConfig,
    projection_files: List[str],
    angles_deg: np.ndarray,
    det_h: int,
    det_w: int,
    output_slice_dir: str,
    voxel_x: float,
    voxel_y: float,
    voxel_z: float,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    n_proj = len(projection_files)
    nx = int(config.recon_nx)
    ny = int(config.recon_ny)
    nz = int(config.recon_nz)
    algo_key = _algorithm_key(config.algorithm)
    iter_count = int(config.iterative_iterations)

    projections = _load_projections_stack(
        projection_files=projection_files,
        det_h=det_h,
        det_w=det_w,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )

    angles_rad = np.deg2rad(angles_deg.astype(np.float32, copy=False))
    proj_geom = _build_astra_proj_geom(
        astra_module=astra_module,
        config=config,
        det_h=det_h,
        det_w=det_w,
        angles_rad=angles_rad,
    )
    vol_geom = _build_astra_volume_geom(
        astra_module=astra_module,
        nx=nx,
        ny=ny,
        nz=nz,
        voxel_x_mm=voxel_x,
        voxel_y_mm=voxel_y,
        voxel_z_mm=voxel_z,
    )

    _emit_progress(progress_callback, 40, 100, "ASTRA: creating CUDA reconstruction job...")
    sino_id = None
    rec_id = None
    fdk_alg_id = None
    iter_alg_id = None
    volume = None
    try:
        _emit_progress(progress_callback, 42, 100, "ASTRA: creating sinogram object...")
        sino_id = astra_module.data3d.create("-sino", proj_geom, projections)
        _emit_progress(progress_callback, 46, 100, "ASTRA: creating volume object...")
        rec_id = astra_module.data3d.create("-vol", vol_geom)
        _emit_progress(progress_callback, 49, 100, "ASTRA: CUDA objects ready.")

        option_base = _get_astra_option_base()
        if algo_key == "fdk":
            cfg = astra_module.astra_dict("FDK_CUDA")
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            option = dict(option_base)
            option["FilterType"] = _astra_filter_name(config.filter_name)
            cfg["option"] = option
            fdk_alg_id = astra_module.algorithm.create(cfg)
            _emit_progress(progress_callback, 55, 100, "ASTRA: running FDK_CUDA...")
            astra_module.algorithm.run(fdk_alg_id)
            backend = "astra_fdk_cuda"

        elif algo_key == "sirt":
            cfg = astra_module.astra_dict("SIRT3D_CUDA")
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            cfg["option"] = dict(option_base)
            iter_alg_id = astra_module.algorithm.create(cfg)
            _emit_progress(
                progress_callback,
                55,
                100,
                f"ASTRA: running SIRT3D_CUDA ({iter_count} iterations)...",
            )
            astra_module.algorithm.run(iter_alg_id, int(max(1, iter_count)))
            backend = "astra_sirt3d_cuda"

        elif algo_key == "cgls":
            cfg = astra_module.astra_dict("CGLS3D_CUDA")
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id
            cfg["option"] = dict(option_base)
            iter_alg_id = astra_module.algorithm.create(cfg)
            _emit_progress(
                progress_callback,
                55,
                100,
                f"ASTRA: running CGLS3D_CUDA ({iter_count} iterations)...",
            )
            astra_module.algorithm.run(iter_alg_id, int(max(1, iter_count)))
            backend = "astra_cgls3d_cuda"

        elif algo_key == "fdk_cgls":
            cfg_fdk = astra_module.astra_dict("FDK_CUDA")
            cfg_fdk["ProjectionDataId"] = sino_id
            cfg_fdk["ReconstructionDataId"] = rec_id
            option_fdk = dict(option_base)
            option_fdk["FilterType"] = _astra_filter_name(config.filter_name)
            cfg_fdk["option"] = option_fdk
            fdk_alg_id = astra_module.algorithm.create(cfg_fdk)
            _emit_progress(progress_callback, 50, 100, "ASTRA: running initial FDK_CUDA...")
            astra_module.algorithm.run(fdk_alg_id)

            cfg_iter = astra_module.astra_dict("CGLS3D_CUDA")
            cfg_iter["ProjectionDataId"] = sino_id
            cfg_iter["ReconstructionDataId"] = rec_id
            cfg_iter["option"] = dict(option_base)
            iter_alg_id = astra_module.algorithm.create(cfg_iter)
            _emit_progress(
                progress_callback,
                65,
                100,
                f"ASTRA: running CGLS3D_CUDA refine ({iter_count} iterations)...",
            )
            astra_module.algorithm.run(iter_alg_id, int(max(1, iter_count)))
            backend = "astra_fdk_cgls3d_cuda"

        else:
            raise ValueError(f"Unsupported reconstruction algorithm: {config.algorithm}")

        _emit_progress(progress_callback, 82, 100, "ASTRA: fetching reconstructed volume...")
        volume = astra_module.data3d.get(rec_id).astype(np.float32, copy=False)

    finally:
        if iter_alg_id is not None:
            astra_module.algorithm.delete(iter_alg_id)
        if fdk_alg_id is not None:
            astra_module.algorithm.delete(fdk_alg_id)
        if rec_id is not None:
            astra_module.data3d.delete(rec_id)
        if sino_id is not None:
            astra_module.data3d.delete(sino_id)

    if volume is None:
        raise RuntimeError("ASTRA reconstruction did not produce valid output.")

    _apply_post_refine_if_needed(
        volume=volume,
        config=config,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )
    _write_volume_slices(
        volume=volume,
        output_slice_dir=output_slice_dir,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )

    return {
        "backend": backend,
        "z_batch_size": int(volume.shape[0]),
        "astra_filter": _astra_filter_name(config.filter_name),
        "algorithm_key": algo_key,
        "iterative_iterations_used": int(iter_count if _is_iterative_algorithm(config.algorithm) else 0),
    }


def _run_fdk_reconstruction_cpu(
    config: ReconstructionConfig,
    projection_files: List[str],
    angles_deg: np.ndarray,
    det_h: int,
    det_w: int,
    output_slice_dir: str,
    voxel_x: float,
    voxel_y: float,
    voxel_z: float,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    nx = int(config.recon_nx)
    ny = int(config.recon_ny)
    nz = int(config.recon_nz)
    du = float(config.detector_pixel_size_x_mm)
    dv = float(config.detector_pixel_size_y_mm)
    sod = float(config.sod_mm)
    sdd = float(config.sdd_mm)
    n_proj = len(projection_files)

    x = (np.arange(nx, dtype=np.float32) - (nx - 1) * 0.5) * voxel_x
    y = (np.arange(ny, dtype=np.float32) - (ny - 1) * 0.5) * voxel_y
    z = (np.arange(nz, dtype=np.float32) - (nz - 1) * 0.5) * voxel_z
    xg, yg = np.meshgrid(x, y, indexing="xy")

    u0 = (det_w - 1) * 0.5 + float(config.cor_offset_px)
    v0 = (det_h - 1) * 0.5

    bytes_per_slice = int(nx * ny * 4)
    target_batch_bytes = 512 * 1024 * 1024
    z_batch_size = max(1, min(nz, target_batch_bytes // max(1, bytes_per_slice)))
    batches = (nz + z_batch_size - 1) // z_batch_size

    _emit_progress(progress_callback, 2, 100, "CPU: running numpy FDK (batched)...")
    for b in range(batches):
        if stop_requested and stop_requested():
            raise RuntimeError("Reconstruction cancelled by user.")

        z_start = b * z_batch_size
        z_end = min(nz, z_start + z_batch_size)
        z_vals = z[z_start:z_end]
        vol_batch = np.zeros((z_end - z_start, ny, nx), dtype=np.float32)

        for p_idx, (path, angle_deg) in enumerate(zip(projection_files, angles_deg.tolist())):
            if stop_requested and stop_requested():
                raise RuntimeError("Reconstruction cancelled by user.")

            proj = _load_projection(path)
            if proj.shape != (det_h, det_w):
                raise ValueError(
                    f"Projection shape mismatch: {os.path.basename(path)} -> {proj.shape}, expected {(det_h, det_w)}"
                )
            proj_f = _filter_projection_fdk(
                proj,
                sdd_mm=sdd,
                pixel_x_mm=du,
                pixel_y_mm=dv,
                filter_name=config.filter_name,
            )

            beta = np.deg2rad(float(angle_deg))
            cb = float(np.cos(beta))
            sb = float(np.sin(beta))

            x_beta = xg * cb + yg * sb
            denom = sod - x_beta
            valid = denom > 1e-6
            lam = np.where(valid, sdd / denom, 0.0).astype(np.float32, copy=False)
            u_phys = (-xg * sb + yg * cb) * lam
            u_idx = (u_phys / du + u0).astype(np.float32, copy=False)
            weight_bp = (lam * lam).astype(np.float32, copy=False)

            for zi, z_mm in enumerate(z_vals.tolist()):
                v_idx = (z_mm * lam / dv + v0).astype(np.float32, copy=False)
                sample = _bilinear_sample(proj_f, v_idx, u_idx)
                vol_batch[zi] += sample * weight_bp

            done_ratio = (b * n_proj + p_idx + 1) / float(max(1, batches * n_proj))
            done = 2 + int(round(done_ratio * 90.0))
            if p_idx == 0 or (p_idx + 1) % 10 == 0 or (p_idx + 1) == n_proj:
                _emit_progress(
                    progress_callback,
                    min(95, done),
                    100,
                    f"CPU: batch {b + 1}/{batches}, projection {p_idx + 1}/{n_proj}",
                )

        vol_batch *= np.float32(np.pi / float(max(1, n_proj)))

        if int(config.refine_iterations) > 0:
            for zi in range(vol_batch.shape[0]):
                vol_batch[zi] = _refine_slice_diffusion(
                    vol_batch[zi],
                    iterations=int(config.refine_iterations),
                    step=float(config.refine_step),
                )

        for zi, z_global in enumerate(range(z_start, z_end)):
            out_path = os.path.join(output_slice_dir, f"slice_{z_global:05d}.tif")
            tiff.imwrite(out_path, vol_batch[zi].astype(np.float32, copy=False), dtype=np.float32)

    return {
        "backend": "cpu_numpy_fdk",
        "z_batch_size": int(z_batch_size),
        "algorithm_key": "fdk",
        "iterative_iterations_used": 0,
    }


def _reconstruct_single_slice_fdk_cpu(
    config: ReconstructionConfig,
    projection_files: List[str],
    angles_deg: np.ndarray,
    det_h: int,
    det_w: int,
    voxel_x: float,
    voxel_y: float,
    voxel_z: float,
    z_index: int,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    nx = int(config.recon_nx)
    ny = int(config.recon_ny)
    nz = int(config.recon_nz)
    if z_index < 0 or z_index >= nz:
        raise ValueError(f"Preview z_index out of range: {z_index}, valid [0, {max(0, nz - 1)}].")

    du = float(config.detector_pixel_size_x_mm)
    dv = float(config.detector_pixel_size_y_mm)
    sod = float(config.sod_mm)
    sdd = float(config.sdd_mm)
    n_proj = len(projection_files)

    x = (np.arange(nx, dtype=np.float32) - (nx - 1) * 0.5) * float(voxel_x)
    y = (np.arange(ny, dtype=np.float32) - (ny - 1) * 0.5) * float(voxel_y)
    z_vals = (np.arange(nz, dtype=np.float32) - (nz - 1) * 0.5) * float(voxel_z)
    z_mm = float(z_vals[int(z_index)])
    xg, yg = np.meshgrid(x, y, indexing="xy")

    u0 = (det_w - 1) * 0.5 + float(config.cor_offset_px)
    v0 = (det_h - 1) * 0.5

    out = np.zeros((ny, nx), dtype=np.float32)
    _emit_progress(
        progress_callback,
        2,
        100,
        f"Preview: reconstructing single slice z={int(z_index)} with CPU FDK...",
    )
    for p_idx, (path, angle_deg) in enumerate(zip(projection_files, angles_deg.tolist())):
        if stop_requested and stop_requested():
            raise RuntimeError("Reconstruction preview cancelled by user.")

        proj = _load_projection(path)
        if proj.shape != (det_h, det_w):
            raise ValueError(
                f"Projection shape mismatch: {os.path.basename(path)} -> {proj.shape}, expected {(det_h, det_w)}"
            )
        proj_f = _filter_projection_fdk(
            proj,
            sdd_mm=sdd,
            pixel_x_mm=du,
            pixel_y_mm=dv,
            filter_name=config.filter_name,
        )

        beta = np.deg2rad(float(angle_deg))
        cb = float(np.cos(beta))
        sb = float(np.sin(beta))

        x_beta = xg * cb + yg * sb
        denom = sod - x_beta
        valid = denom > 1e-6
        lam = np.where(valid, sdd / denom, 0.0).astype(np.float32, copy=False)
        u_phys = (-xg * sb + yg * cb) * lam
        u_idx = (u_phys / du + u0).astype(np.float32, copy=False)
        v_idx = (z_mm * lam / dv + v0).astype(np.float32, copy=False)
        weight_bp = (lam * lam).astype(np.float32, copy=False)

        sample = _bilinear_sample(proj_f, v_idx, u_idx)
        out += sample * weight_bp

        if p_idx == 0 or (p_idx + 1) % 10 == 0 or (p_idx + 1) == n_proj:
            done = 2 + int(round((p_idx + 1) / max(1, n_proj) * 92.0))
            _emit_progress(
                progress_callback,
                min(95, done),
                100,
                f"Preview: projection {p_idx + 1}/{n_proj}",
            )

    out *= np.float32(np.pi / float(max(1, n_proj)))
    if int(config.refine_iterations) > 0:
        _emit_progress(progress_callback, 96, 100, "Preview: applying post-refine diffusion...")
        out = _refine_slice_diffusion(
            out,
            iterations=int(config.refine_iterations),
            step=float(config.refine_step),
        )
    return out.astype(np.float32, copy=False)


def run_reconstruction_preview_slice(
    config: ReconstructionConfig,
    projection_files: List[str],
    z_index: Optional[int] = None,
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> ReconstructionPreviewResult:
    validate_config(config)
    derived, angles_deg = build_stage1_plan(config)
    if len(projection_files) != int(config.projection_count):
        raise ValueError("Projection file count does not match config.projection_count.")

    _emit_progress(progress_callback, 0, 100, "Checking projection data for preview...")
    first = _load_projection(projection_files[0])
    det_h, det_w = first.shape
    nz = int(config.recon_nz)
    if nz < 1:
        raise ValueError("Invalid reconstruction depth: recon_nz must be >= 1.")

    if z_index is None:
        z_index = nz // 2
    z_index = int(np.clip(int(z_index), 0, nz - 1))

    preview_dir = os.path.join(config.output_folder, "proview")
    os.makedirs(preview_dir, exist_ok=True)

    backend_note = "Preview uses CPU single-slice FDK reconstruction."
    if _is_iterative_algorithm(config.algorithm):
        backend_note += " Iterative algorithm settings are ignored in preview."
    _emit_progress(progress_callback, 1, 100, backend_note)

    t0 = time.time()
    preview_slice = _reconstruct_single_slice_fdk_cpu(
        config=config,
        projection_files=projection_files,
        angles_deg=angles_deg,
        det_h=det_h,
        det_w=det_w,
        voxel_x=float(derived.voxel_size_x_mm),
        voxel_y=float(derived.voxel_size_y_mm),
        voxel_z=float(derived.voxel_size_z_mm),
        z_index=int(z_index),
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )

    if stop_requested and stop_requested():
        raise RuntimeError("Reconstruction preview cancelled by user.")

    preview_path = os.path.join(preview_dir, f"preview_z{int(z_index):05d}.tif")
    tiff.imwrite(preview_path, preview_slice.astype(np.float32, copy=False), dtype=np.float32)
    elapsed = float(time.time() - t0)
    _emit_progress(progress_callback, 100, 100, f"Preview slice saved: {preview_path}")
    return ReconstructionPreviewResult(
        preview_path=preview_path,
        preview_dir=preview_dir,
        z_index=int(z_index),
        elapsed_seconds=elapsed,
        backend_used="cpu_numpy_fdk_single_slice",
        backend_note=backend_note,
    )


def run_fdk_reconstruction(
    config: ReconstructionConfig,
    projection_files: List[str],
    progress_callback: ProgressCallback = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> ReconstructionRunResult:
    validate_config(config)
    derived, angles_deg = build_stage1_plan(config)
    if len(projection_files) != int(config.projection_count):
        raise ValueError("Projection file count does not match config.projection_count.")

    _emit_progress(progress_callback, 0, 100, "Checking projection data...")
    first = _load_projection(projection_files[0])
    det_h, det_w = first.shape

    output_slice_dir = os.path.join(config.output_folder, "recon_slices")
    os.makedirs(output_slice_dir, exist_ok=True)

    angle_csv_path = os.path.join(config.output_folder, "reconstruction_angles.csv")
    with open(angle_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "angle_deg"])
        for i, angle in enumerate(angles_deg.tolist()):
            writer.writerow([i, float(angle)])

    stage1_plan_path = os.path.join(config.output_folder, "reconstruction_stage1_plan.json")
    stage1_payload = {
        "stage": "phase3_execute",
        "projection_folder": config.projection_folder,
        "output_folder": config.output_folder,
        "projection_count": int(config.projection_count),
        "projection_shape_hw": [int(det_h), int(det_w)],
        "algorithm": config.algorithm,
        "iterative_iterations": int(config.iterative_iterations),
        "filter_name": config.filter_name,
        "geometry": {
            "sod_mm": float(config.sod_mm),
            "sdd_mm": float(config.sdd_mm),
            "detector_pixel_size_x_mm": float(config.detector_pixel_size_x_mm),
            "detector_pixel_size_y_mm": float(config.detector_pixel_size_y_mm),
            "cor_offset_px": float(config.cor_offset_px),
        },
        "angles": {
            "rule": "theta_i = start_angle_deg + i * angle_step_deg",
            "start_angle_deg": float(config.start_angle_deg),
            "angle_step_deg": float(config.angle_step_deg),
            "count": int(config.projection_count),
            "first_angle_deg": float(angles_deg[0]),
            "last_angle_deg": float(angles_deg[-1]),
            "total_scan_angle_deg": float(derived.total_scan_angle_deg),
            "csv_path": angle_csv_path,
        },
        "volume": {
            "nx": int(config.recon_nx),
            "ny": int(config.recon_ny),
            "nz": int(config.recon_nz),
            "voxel_size_x_mm": float(derived.voxel_size_x_mm),
            "voxel_size_y_mm": float(derived.voxel_size_y_mm),
            "voxel_size_z_mm": float(derived.voxel_size_z_mm),
        },
        "post_refine": {
            "iterations": int(config.refine_iterations),
            "step": float(config.refine_step),
            "method": "slice_laplacian_diffusion",
        },
        "output_format": config.output_format,
        "backend_policy": "prefer_astra_cuda_then_cpu_fallback",
        "note": "Stage-3 reconstruction: prefer ASTRA CUDA; fallback to CPU numpy FDK when unavailable.",
    }
    with open(stage1_plan_path, "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    nx = int(config.recon_nx)
    ny = int(config.recon_ny)
    nz = int(config.recon_nz)
    voxel_x = float(derived.voxel_size_x_mm)
    voxel_y = float(derived.voxel_size_y_mm)
    voxel_z = float(derived.voxel_size_z_mm)
    n_proj = len(projection_files)
    algo_key = _algorithm_key(config.algorithm)

    t0 = time.time()
    backend_used = "cpu_numpy_fdk"
    backend_note = ""
    z_batch_size = 0
    iterative_iterations_used = 0

    astra_module, astra_import_error = _try_import_astra()
    force_cpu = os.environ.get("RECON_FORCE_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    est_astra_bytes = _estimate_astra_required_bytes(
        det_h=det_h,
        det_w=det_w,
        n_proj=n_proj,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    free_cuda_bytes = _get_cuda_free_bytes()
    allow_astra = True
    if force_cpu and astra_module is not None:
        allow_astra = False
        backend_note = "RECON_FORCE_CPU enabled; skip ASTRA and use CPU batched reconstruction."
        _emit_progress(progress_callback, 2, 100, backend_note)

    if astra_module is not None and free_cuda_bytes is not None:
        projection_bytes = int(det_h) * int(det_w) * int(n_proj) * 4
        try:
            safe_ratio = float(os.environ.get("RECON_ASTRA_SAFE_RATIO", "0.60"))
        except Exception:
            safe_ratio = 0.60
        safe_ratio = max(0.1, min(safe_ratio, 0.95))

        try:
            proj_ratio = float(os.environ.get("RECON_ASTRA_PROJ_RATIO", "0.55"))
        except Exception:
            proj_ratio = 0.55
        proj_ratio = max(0.1, min(proj_ratio, 0.95))

        if (est_astra_bytes > int(free_cuda_bytes * safe_ratio)) or (
            projection_bytes > int(free_cuda_bytes * proj_ratio)
        ):
            allow_astra = False
            need_gb = est_astra_bytes / (1024 ** 3)
            free_gb = free_cuda_bytes / (1024 ** 3)
            proj_gb = projection_bytes / (1024 ** 3)
            backend_note = (
                f"Estimated ASTRA memory need ~ {need_gb:.1f} GB "
                f"(projection stack ~ {proj_gb:.1f} GB), available ~ {free_gb:.1f} GB; "
                "skip ASTRA and use CPU batched reconstruction."
            )
            _emit_progress(progress_callback, 2, 100, backend_note)
    if astra_module is not None and allow_astra:
        try:
            backend_info = _run_astra_reconstruction(
                astra_module=astra_module,
                config=config,
                projection_files=projection_files,
                angles_deg=angles_deg,
                det_h=det_h,
                det_w=det_w,
                output_slice_dir=output_slice_dir,
                voxel_x=voxel_x,
                voxel_y=voxel_y,
                voxel_z=voxel_z,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
            )
            backend_used = str(backend_info.get("backend", "astra_fdk_cuda"))
            z_batch_size = int(backend_info.get("z_batch_size", nz))
            iterative_iterations_used = int(backend_info.get("iterative_iterations_used", 0))
        except Exception as e:
            if stop_requested and stop_requested():
                raise RuntimeError("Reconstruction cancelled by user.") from e
            backend_note = f"ASTRA failed, falling back to CPU FDK: {e}"
            _emit_progress(progress_callback, 2, 100, backend_note)
            backend_info = _run_fdk_reconstruction_cpu(
                config=config,
                projection_files=projection_files,
                angles_deg=angles_deg,
                det_h=det_h,
                det_w=det_w,
                output_slice_dir=output_slice_dir,
                voxel_x=voxel_x,
                voxel_y=voxel_y,
                voxel_z=voxel_z,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
            )
            backend_used = str(backend_info.get("backend", "cpu_numpy_fdk"))
            z_batch_size = int(backend_info.get("z_batch_size", 0))
            iterative_iterations_used = int(backend_info.get("iterative_iterations_used", 0))
    elif astra_module is None:
        if _is_iterative_algorithm(config.algorithm):
            backend_note = (
                "ASTRA unavailable; iterative algorithm cannot run, fallback to CPU numpy FDK. "
                f" import_error={astra_import_error or 'unknown'}"
            )
        else:
            backend_note = f"ASTRA unavailable, using CPU numpy FDK: {astra_import_error or 'unknown'}"
        _emit_progress(progress_callback, 2, 100, backend_note)
        backend_info = _run_fdk_reconstruction_cpu(
            config=config,
            projection_files=projection_files,
            angles_deg=angles_deg,
            det_h=det_h,
            det_w=det_w,
            output_slice_dir=output_slice_dir,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            voxel_z=voxel_z,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
        backend_used = str(backend_info.get("backend", "cpu_numpy_fdk"))
        z_batch_size = int(backend_info.get("z_batch_size", 0))
        iterative_iterations_used = int(backend_info.get("iterative_iterations_used", 0))
    else:
        backend_info = _run_fdk_reconstruction_cpu(
            config=config,
            projection_files=projection_files,
            angles_deg=angles_deg,
            det_h=det_h,
            det_w=det_w,
            output_slice_dir=output_slice_dir,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            voxel_z=voxel_z,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
        backend_used = str(backend_info.get("backend", "cpu_numpy_fdk"))
        z_batch_size = int(backend_info.get("z_batch_size", 0))
        iterative_iterations_used = int(backend_info.get("iterative_iterations_used", 0))
    elapsed = float(time.time() - t0)
    summary_path = os.path.join(config.output_folder, "run_summary_recon.json")
    summary_payload = {
        "backend_used": backend_used,
        "backend_note": backend_note,
        "algorithm": config.algorithm,
        "algorithm_key": algo_key,
        "iterative_iterations_config": int(config.iterative_iterations),
        "iterative_iterations_used": int(iterative_iterations_used),
        "filter_name": config.filter_name,
        "projection_count": int(n_proj),
        "projection_shape_hw": [int(det_h), int(det_w)],
        "volume_shape_xyz": [int(nx), int(ny), int(nz)],
        "voxel_size_mm": [float(voxel_x), float(voxel_y), float(voxel_z)],
        "z_batch_size": int(z_batch_size),
        "post_refine_iterations": int(config.refine_iterations),
        "post_refine_step": float(config.refine_step),
        "slice_output_dir": output_slice_dir,
        "angles_csv_path": angle_csv_path,
        "stage1_plan_json_path": stage1_plan_path,
        "elapsed_seconds": elapsed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    _emit_progress(progress_callback, 100, 100, "Reconstruction completed.")
    return ReconstructionRunResult(
        output_slice_dir=output_slice_dir,
        summary_json_path=summary_path,
        angles_csv_path=angle_csv_path,
        stage1_plan_json_path=stage1_plan_path,
        slice_count=nz,
        projection_count=n_proj,
        elapsed_seconds=elapsed,
        backend_used=backend_used,
        backend_note=backend_note,
    )

