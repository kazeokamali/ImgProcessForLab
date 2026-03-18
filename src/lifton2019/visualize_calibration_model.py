import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


@dataclass
class ModelData:
    x_knots: np.ndarray
    y_refs: np.ndarray
    dark_knots: Optional[np.ndarray]
    flat_knots: Optional[np.ndarray]
    summary: Optional[dict]


def _load_npy(path: Path, required: bool = True) -> Optional[np.ndarray]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少文件: {path}")
        return None
    return np.load(str(path))


def load_model(model_dir: Path) -> ModelData:
    x_knots = _load_npy(model_dir / "calibration_x_knots.npy", required=True)
    y_refs = _load_npy(model_dir / "calibration_y_refs.npy", required=True)
    dark_knots = _load_npy(model_dir / "calibration_dark_knots.npy", required=False)
    flat_knots = _load_npy(model_dir / "calibration_flat_knots.npy", required=False)

    summary_path = model_dir / "calibration_model_summary.json"
    summary = None
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    if x_knots.ndim != 3:
        raise ValueError(f"x_knots 维度错误，期望 3 维，实际 {x_knots.ndim}")
    if y_refs.ndim != 1:
        raise ValueError(f"y_refs 维度错误，期望 1 维，实际 {y_refs.ndim}")
    if x_knots.shape[0] != y_refs.shape[0]:
        raise ValueError(
            f"x_knots 点数({x_knots.shape[0]}) 与 y_refs 点数({y_refs.shape[0]}) 不一致"
        )
    if dark_knots is not None and dark_knots.shape != x_knots.shape:
        raise ValueError(
            f"dark_knots 形状错误，期望 {x_knots.shape}，实际 {dark_knots.shape}"
        )
    if flat_knots is not None and flat_knots.shape != x_knots.shape:
        raise ValueError(
            f"flat_knots 形状错误，期望 {x_knots.shape}，实际 {flat_knots.shape}"
        )

    return ModelData(
        x_knots=x_knots.astype(np.float32, copy=False),
        y_refs=y_refs.astype(np.float32, copy=False),
        dark_knots=dark_knots.astype(np.float32, copy=False) if dark_knots is not None else None,
        flat_knots=flat_knots.astype(np.float32, copy=False) if flat_knots is not None else None,
        summary=summary,
    )


def robust_range(arr: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(valid, q_low))
    vmax = float(np.percentile(valid, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def save_heatmap(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def save_curve(y: np.ndarray, x: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def write_stats_report(data: ModelData, out_path: Path):
    x = data.x_knots
    y = data.y_refs
    report = {
        "x_knots_shape": list(x.shape),
        "y_refs_shape": list(y.shape),
        "x_knots_dtype": str(x.dtype),
        "y_refs_dtype": str(y.dtype),
        "x_knots_min": float(np.nanmin(x)),
        "x_knots_max": float(np.nanmax(x)),
        "x_knots_mean": float(np.nanmean(x)),
        "y_refs_values": [float(v) for v in y.tolist()],
        "finite_ratio_x": float(np.isfinite(x).mean()),
        "finite_ratio_y": float(np.isfinite(y).mean()),
    }

    x_diff = np.diff(x.astype(np.float64), axis=0)
    report["x_monotonic_violations"] = int((x_diff < -1e-6).sum())
    y_diff = np.diff(y.astype(np.float64))
    report["y_monotonic_violations"] = int((y_diff < -1e-9).sum())

    if data.dark_knots is not None:
        d = data.dark_knots
        report["dark_knots_shape"] = list(d.shape)
        report["dark_knots_min"] = float(np.nanmin(d))
        report["dark_knots_max"] = float(np.nanmax(d))
        report["dark_knots_mean"] = float(np.nanmean(d))
    if data.flat_knots is not None:
        f = data.flat_knots
        report["flat_knots_shape"] = list(f.shape)
        report["flat_knots_min"] = float(np.nanmin(f))
        report["flat_knots_max"] = float(np.nanmax(f))
        report["flat_knots_mean"] = float(np.nanmean(f))
    if data.dark_knots is not None and data.flat_knots is not None:
        err = np.abs((data.flat_knots - data.dark_knots) - data.x_knots)
        report["consistency_error_max"] = float(np.nanmax(err))
        report["consistency_error_mean"] = float(np.nanmean(err))

    if data.summary is not None:
        report["summary_json"] = data.summary

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def sample_pixel_indices(height: int, width: int):
    rows = [0, height // 4, height // 2, (3 * height) // 4, height - 1]
    cols = [0, width // 4, width // 2, (3 * width) // 4, width - 1]
    points = []
    for r in rows:
        for c in cols:
            points.append((int(r), int(c)))
    return points


def save_sample_curves(data: ModelData, out_dir: Path):
    x = data.x_knots.astype(np.float64)
    y = data.y_refs.astype(np.float64)
    _, h, w = x.shape
    pts = sample_pixel_indices(h, w)

    plt.figure(figsize=(12, 8))
    for r, c in pts:
        plt.plot(y, x[:, r, c], marker="o", linewidth=1.0, markersize=2.5, label=f"({r},{c})")
    plt.title("Sample Pixel Response Curves: y_refs -> x_knots")
    plt.xlabel("y_refs")
    plt.ylabel("x_knots")
    plt.grid(True, alpha=0.3)
    if len(pts) <= 25:
        plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(str(out_dir / "sample_pixel_curves.png"), dpi=160)
    plt.close()


def save_point_maps(name: str, arr: np.ndarray, out_dir: Path, save_tiff: bool = True):
    n = arr.shape[0]
    vmin, vmax = robust_range(arr.astype(np.float64), q_low=1.0, q_high=99.0)
    for i in range(n):
        img = arr[i]
        save_heatmap(
            img,
            out_dir / f"{name}_{i:03d}.png",
            title=f"{name}[{i}]",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        if save_tiff:
            tiff.imwrite(str(out_dir / f"{name}_{i:03d}.tif"), img.astype(np.float32), dtype=np.float32)


def save_overview_maps(data: ModelData, out_dir: Path):
    x = data.x_knots.astype(np.float64)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    save_heatmap(x_mean, out_dir / "x_knots_mean_map.png", "x_knots Mean Map")
    save_heatmap(x_std, out_dir / "x_knots_std_map.png", "x_knots Std Map")

    x_diff = np.diff(x, axis=0)
    vio_count = np.sum(x_diff < -1e-6, axis=0).astype(np.int32)
    save_heatmap(
        vio_count.astype(np.float32),
        out_dir / "x_knots_monotonic_violation_count.png",
        "x_knots Monotonic Violation Count",
        cmap="magma",
    )

    if data.dark_knots is not None and data.flat_knots is not None:
        err = np.abs((data.flat_knots.astype(np.float64) - data.dark_knots.astype(np.float64)) - x)
        err_mean = np.mean(err, axis=0)
        err_max = np.max(err, axis=0)
        save_heatmap(err_mean, out_dir / "consistency_error_mean_map.png", "Consistency Error Mean |flat-dark-x|")
        save_heatmap(err_max, out_dir / "consistency_error_max_map.png", "Consistency Error Max |flat-dark-x|")


def visualize_model(model_dir: Path, output_dir: Path, save_tiff: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    knots_dir = output_dir / "knots"
    knots_dir.mkdir(parents=True, exist_ok=True)

    data = load_model(model_dir)

    write_stats_report(data, output_dir / "model_stats_report.json")

    save_curve(
        y=data.y_refs.astype(np.float64),
        x=np.arange(len(data.y_refs)),
        out_path=output_dir / "y_refs_by_point_index.png",
        title="y_refs by Point Index",
        xlabel="point index",
        ylabel="y_refs",
    )

    save_point_maps("x_knots", data.x_knots, knots_dir, save_tiff=save_tiff)
    if data.dark_knots is not None:
        save_point_maps("dark_knots", data.dark_knots, knots_dir, save_tiff=save_tiff)
    if data.flat_knots is not None:
        save_point_maps("flat_knots", data.flat_knots, knots_dir, save_tiff=save_tiff)

    save_overview_maps(data, output_dir)
    save_sample_curves(data, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="可视化平场标定模型 .npy 文件并保存图像")
    parser.add_argument("--model-dir", required=True, help="模型目录，包含 calibration_*.npy")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认: <model-dir>/visualization",
    )
    parser.add_argument(
        "--no-tiff",
        action="store_true",
        help="不额外导出每个点位的 .tif 原始图",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else (model_dir / "visualization")
    visualize_model(model_dir=model_dir, output_dir=output_dir, save_tiff=(not args.no_tiff))
    print(f"可视化完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
