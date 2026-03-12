import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, grey_opening, map_coordinates


ArrayLike = np.ndarray
CenterXY = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class PolarMeta:
    shape: Tuple[int, int]
    center_y: float
    center_x: float
    num_angles: int
    num_radii: int
    max_radius: float


class PolarTransformer:
    """Cache-heavy polar/cartesian transformer for batch slice processing."""

    def __init__(self):
        self._cartesian_to_polar_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._polar_to_cartesian_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def cartesian_to_polar(
        self,
        image: ArrayLike,
        center_xy: CenterXY = None,
        num_angles: int = 2048,
        num_radii: Optional[int] = None,
        interpolation_order: int = 1,
    ) -> Tuple[ArrayLike, PolarMeta]:
        img = _as_float32_image(image)
        height, width = img.shape

        center_y, center_x = _sanitize_center((height, width), center_xy)
        max_radius = _max_inscribed_radius((height, width), center_y, center_x)

        if num_radii is None:
            num_radii = int(math.ceil(max_radius)) + 1

        num_angles = max(int(num_angles), 360)
        num_radii = max(int(num_radii), 16)

        cache_key = (
            height,
            width,
            round(center_y, 4),
            round(center_x, 4),
            num_angles,
            num_radii,
            round(max_radius, 4),
        )

        if cache_key not in self._cartesian_to_polar_cache:
            theta = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float32)
            radius = np.linspace(0.0, max_radius, num_radii, dtype=np.float32)

            theta_grid = theta[:, None]
            radius_grid = radius[None, :]

            y = center_y + radius_grid * np.sin(theta_grid)
            x = center_x + radius_grid * np.cos(theta_grid)
            coords = np.vstack((y.ravel(), x.ravel()))
            self._cartesian_to_polar_cache[cache_key] = (coords, np.array([num_angles, num_radii], dtype=np.int32))

        coords, target_shape = self._cartesian_to_polar_cache[cache_key]
        polar = map_coordinates(
            img,
            coords,
            order=interpolation_order,
            mode="reflect",
        ).reshape(int(target_shape[0]), int(target_shape[1]))

        meta = PolarMeta(
            shape=(height, width),
            center_y=float(center_y),
            center_x=float(center_x),
            num_angles=int(target_shape[0]),
            num_radii=int(target_shape[1]),
            max_radius=float(max_radius),
        )
        return polar.astype(np.float32, copy=False), meta

    def polar_to_cartesian(
        self,
        polar_image: ArrayLike,
        meta: PolarMeta,
        interpolation_order: int = 1,
        reference_image: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        polar = np.asarray(polar_image, dtype=np.float32)
        height, width = meta.shape

        cache_key = (
            height,
            width,
            round(meta.center_y, 4),
            round(meta.center_x, 4),
            meta.num_angles,
            meta.num_radii,
            round(meta.max_radius, 4),
        )

        if cache_key not in self._polar_to_cartesian_cache:
            yy, xx = np.indices((height, width), dtype=np.float32)
            dy = yy - meta.center_y
            dx = xx - meta.center_x
            radius = np.sqrt(dx * dx + dy * dy)

            theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
            theta_idx = theta / (2.0 * np.pi) * float(meta.num_angles)
            radius_idx = radius / max(meta.max_radius, 1e-6) * float(meta.num_radii - 1)
            outside_mask = radius > meta.max_radius

            coords = np.vstack((theta_idx.ravel(), radius_idx.ravel()))
            self._polar_to_cartesian_cache[cache_key] = (coords, outside_mask, np.array([height, width], dtype=np.int32))

        coords, outside_mask, target_shape = self._polar_to_cartesian_cache[cache_key]

        # Duplicate first angle row to guarantee periodic interpolation continuity.
        polar_extended = np.vstack((polar, polar[:1, :]))
        cartesian = map_coordinates(
            polar_extended,
            coords,
            order=interpolation_order,
            mode="nearest",
        ).reshape(int(target_shape[0]), int(target_shape[1]))

        if reference_image is not None:
            ref = np.asarray(reference_image, dtype=np.float32)
            cartesian[outside_mask] = ref[outside_mask]
        else:
            cartesian[outside_mask] = 0.0

        return cartesian.astype(np.float32, copy=False)


def remove_ring_artifact_polar(
    image: ArrayLike,
    *,
    center_xy: CenterXY = None,
    num_angles: int = 2048,
    stripe_sigma: float = 6.0,
    correction_strength: float = 1.0,
    transformer: Optional[PolarTransformer] = None,
) -> ArrayLike:
    """Classic method: cartesian -> polar -> stripe suppression -> inverse."""
    if transformer is None:
        transformer = PolarTransformer()

    img = _as_float32_image(image)
    polar, meta = transformer.cartesian_to_polar(
        img,
        center_xy=center_xy,
        num_angles=num_angles,
        interpolation_order=1,
    )

    stripe_sigma = max(float(stripe_sigma), 0.1)
    correction_strength = max(float(correction_strength), 0.0)

    radial_profile = np.median(polar, axis=0)
    radial_trend = gaussian_filter1d(radial_profile, sigma=stripe_sigma, mode="nearest")
    stripe_component = radial_profile - radial_trend

    corrected_polar = polar - correction_strength * stripe_component[np.newaxis, :]
    corrected = transformer.polar_to_cartesian(
        corrected_polar,
        meta=meta,
        interpolation_order=1,
        reference_image=img,
    )
    return _preserve_mean(img, corrected)


def remove_ring_artifact_frequency(
    image: ArrayLike,
    *,
    center_xy: CenterXY = None,
    num_angles: int = 2048,
    low_freq_cutoff: int = 3,
    suppression_ratio: float = 0.7,
    periodic_notch: int = 0,
    notch_width: int = 1,
    transformer: Optional[PolarTransformer] = None,
) -> ArrayLike:
    """Frequency-domain suppression in polar domain (angular axis FFT)."""
    if transformer is None:
        transformer = PolarTransformer()

    img = _as_float32_image(image)
    polar, meta = transformer.cartesian_to_polar(
        img,
        center_xy=center_xy,
        num_angles=num_angles,
        interpolation_order=1,
    )

    low_freq_cutoff = max(int(low_freq_cutoff), 0)
    suppression_ratio = float(np.clip(suppression_ratio, 0.0, 1.0))
    periodic_notch = max(int(periodic_notch), 0)
    notch_width = max(int(notch_width), 1)

    spectrum = np.fft.rfft(polar, axis=0)
    max_freq_index = spectrum.shape[0] - 1

    # Keep DC component to avoid global brightness drift.
    if low_freq_cutoff >= 1:
        end_idx = min(low_freq_cutoff, max_freq_index)
        spectrum[1 : end_idx + 1, :] *= (1.0 - suppression_ratio)

    if periodic_notch > 0 and periodic_notch <= max_freq_index:
        left = max(periodic_notch - notch_width, 1)
        right = min(periodic_notch + notch_width, max_freq_index)
        spectrum[left : right + 1, :] *= (1.0 - suppression_ratio)

    corrected_polar = np.fft.irfft(spectrum, n=polar.shape[0], axis=0).astype(np.float32, copy=False)
    corrected = transformer.polar_to_cartesian(
        corrected_polar,
        meta=meta,
        interpolation_order=1,
        reference_image=img,
    )
    return _preserve_mean(img, corrected)


def remove_ring_artifact_morphology(
    image: ArrayLike,
    *,
    center_xy: CenterXY = None,
    num_angles: int = 2048,
    opening_theta_size: int = 101,
    profile_sigma: float = 6.0,
    correction_strength: float = 1.0,
    transformer: Optional[PolarTransformer] = None,
) -> ArrayLike:
    """Morphology-guided stripe estimate in polar domain."""
    if transformer is None:
        transformer = PolarTransformer()

    img = _as_float32_image(image)
    polar, meta = transformer.cartesian_to_polar(
        img,
        center_xy=center_xy,
        num_angles=num_angles,
        interpolation_order=1,
    )

    opening_theta_size = _odd_at_least(opening_theta_size, 3)
    profile_sigma = max(float(profile_sigma), 0.1)
    correction_strength = max(float(correction_strength), 0.0)

    opened = grey_opening(polar, size=(opening_theta_size, 1))
    stripe_profile = np.median(opened, axis=0)
    stripe_trend = gaussian_filter1d(stripe_profile, sigma=profile_sigma, mode="nearest")
    stripe_component = stripe_profile - stripe_trend

    corrected_polar = polar - correction_strength * stripe_component[np.newaxis, :]
    corrected = transformer.polar_to_cartesian(
        corrected_polar,
        meta=meta,
        interpolation_order=1,
        reference_image=img,
    )
    return _preserve_mean(img, corrected)


def _as_float32_image(image: ArrayLike) -> ArrayLike:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("Ring artifact processing only supports 2D grayscale images.")
    return arr.astype(np.float32, copy=False)


def _sanitize_center(shape: Tuple[int, int], center_xy: CenterXY) -> Tuple[float, float]:
    height, width = shape
    if center_xy is None:
        return (height - 1) / 2.0, (width - 1) / 2.0

    x, y = center_xy
    x = float(np.clip(x, 0.0, float(width - 1)))
    y = float(np.clip(y, 0.0, float(height - 1)))
    return y, x


def _max_inscribed_radius(shape: Tuple[int, int], center_y: float, center_x: float) -> float:
    height, width = shape
    return float(min(center_x, center_y, width - 1 - center_x, height - 1 - center_y))


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), int(minimum))
    if value % 2 == 0:
        value += 1
    return value


def _preserve_mean(original: ArrayLike, corrected: ArrayLike) -> ArrayLike:
    delta = float(np.mean(original) - np.mean(corrected))
    output = corrected + delta
    return output.astype(np.float32, copy=False)
