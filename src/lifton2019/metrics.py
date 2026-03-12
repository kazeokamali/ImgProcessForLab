import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def compute_air_roi_mean(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> float:
    if roi is None:
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


def compute_stripe_strength(image: np.ndarray) -> float:
    col_mean = np.mean(image, axis=0)
    return float(np.std(col_mean))


@dataclass
class MetricsCollector:
    rows: List[List[object]] = field(default_factory=list)

    def add(self, frame_index: int, file_name: str, air_roi_mean: float, stripe_strength: float):
        self.rows.append([frame_index, file_name, air_roi_mean, stripe_strength])

    def save_csv(self, path: str):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "file_name", "air_roi_mean", "stripe_strength"])
            writer.writerows(self.rows)

    def summary(self) -> dict:
        if not self.rows:
            return {
                "frame_count": 0,
                "air_roi_mean_avg": None,
                "air_roi_mean_std": None,
                "stripe_strength_avg": None,
                "stripe_strength_std": None,
            }

        air = np.asarray([float(r[2]) for r in self.rows], dtype=np.float64)
        stripe = np.asarray([float(r[3]) for r in self.rows], dtype=np.float64)
        return {
            "frame_count": len(self.rows),
            "air_roi_mean_avg": float(np.mean(air)),
            "air_roi_mean_std": float(np.std(air)),
            "stripe_strength_avg": float(np.mean(stripe)),
            "stripe_strength_std": float(np.std(stripe)),
        }

