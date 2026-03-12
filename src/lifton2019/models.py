from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BadPixelConfig:
    enabled: bool = True
    enable_flat_neighbor_check: bool = True
    enable_dark_neighbor_check: bool = True
    enable_stability_check: bool = True

    known_mask_path: str = ""
    known_badline_path: str = ""

    neighborhood_size: int = 3
    flat_neighbor_sigma: float = 8.0
    dark_neighbor_sigma: float = 8.0
    stability_sigma: float = 6.0

    dilation_radius: int = 0
    min_component_size: int = 1

    repair_window_size: int = 3
    repair_iterations: int = 6
    enable_directional_line_repair: bool = True
    directional_line_aspect_ratio: float = 6.0


@dataclass
class ProcessingConfig:
    num_points: int = 7
    raw_width: int = 2340
    raw_height: int = 2882
    point_pattern: str = r"(\d+)"
    use_roi_reference: bool = False
    reference_roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    epsilon: float = 1e-6
    bad_pixel: BadPixelConfig = field(default_factory=BadPixelConfig)


@dataclass
class CalibrationSet:
    dark_avg: np.ndarray
    flat_avgs: List[np.ndarray]
    flat_refs: np.ndarray
    point_ids: List[str]
    frame_counts: Dict[str, int]
    dark_std: Optional[np.ndarray] = None
    flat_stds: Optional[List[np.ndarray]] = None


@dataclass
class PiecewiseKnots:
    x_knots: np.ndarray  # [N, H, W], dark-corrected flat response per point
    y_knots: np.ndarray  # [N], reference intensities
    point_ids: List[str]


@dataclass
class ProcessingResult:
    processed_count: int
    failed_count: int
    output_folder: str
    metrics_csv_path: str
    summary_json_path: str
