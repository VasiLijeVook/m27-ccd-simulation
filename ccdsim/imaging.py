"""Measurements on astronomical images: scaling, background, sources, FWHM, profiles."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter


def display_limits(img, low=0.5, high=99.7):
    """Percentile limits for plotting, so a few bright pixels do not ruin the contrast."""
    return np.percentile(img, [low, high])


def normalize_image(img, low=5, high=99.5):
    """Scale an image to 0-1 using robust percentiles.

    The lower percentile is subtracted as background and the image is divided by the
    upper percentile, so a single saturated star or cosmic ray cannot squeeze the rest
    of the image towards zero (which happens when dividing by the maximum).
    """
    img = np.nan_to_num(np.asarray(img, dtype=float))
    img = img - np.percentile(img, low)
    top = np.percentile(img, high)
    if top <= 0:
        return np.zeros_like(img)
    return np.clip(img / top, 0, 1)


def estimate_background_noise(img, border_width=100):
    """Sky background (median) and noise (scaled MAD) measured on the image borders,
    which are least affected by the nebula."""
    border = np.concatenate([
        img[:border_width, :].ravel(),
        img[-border_width:, :].ravel(),
        img[:, :border_width].ravel(),
        img[:, -border_width:].ravel(),
    ])
    background = np.median(border)
    noise = 1.4826 * np.median(np.abs(border - background))
    return float(background), float(noise)


def detect_sources(img, background, noise, threshold_sigma=8, max_sharpness=0.5, border=5):
    """Find stars as local maxima above background + threshold_sigma * noise.

    Single-pixel events (hot pixels, cosmic rays) are rejected: for them almost all the
    light in a 3x3 box sits in the central pixel, while a real star is spread by the PSF.
    Returns x and y pixel coordinates, brightest first.
    """
    img = np.asarray(img, dtype=float)
    smoothed = gaussian_filter(img, 1.2)
    peaks = (smoothed == maximum_filter(smoothed, size=9)) & (smoothed > background + threshold_sigma * noise)

    excess = img - background
    box_sum = 9 * uniform_filter(excess, size=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpness = np.where(box_sum > 0, excess / box_sum, 1.0)
    peaks &= sharpness < max_sharpness
    peaks[:border, :] = peaks[-border:, :] = False
    peaks[:, :border] = peaks[:, -border:] = False

    ys, xs = np.nonzero(peaks)
    order = np.argsort(smoothed[ys, xs])[::-1]
    return xs[order], ys[order]


def estimate_fwhm(img, xs, ys, max_sources=200, half_size=15, saturation=60000.0):
    """Median stellar FWHM in pixels, from the area above half maximum.

    For each star, the local sky (median of the cutout border) is subtracted and the
    pixels above half the peak are counted: for a round star that area is
    pi * (FWHM / 2)^2. Unlike second moments, this is not inflated by background noise.
    """
    ny, nx = img.shape
    values = []
    for x, y in zip(xs[:max_sources], ys[:max_sources]):
        if not (half_size <= x < nx - half_size and half_size <= y < ny - half_size):
            continue
        cut = img[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1].astype(float)
        if cut.max() >= saturation:
            continue
        border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
        cut = cut - np.median(border)
        peak = cut[half_size - 1:half_size + 2, half_size - 1:half_size + 2].max()
        if peak <= 0:
            continue
        area = np.count_nonzero(cut >= peak / 2)
        values.append(2 * np.sqrt(area / np.pi))
    return float(np.median(values)) if values else float("nan")


def radial_profile(img, center_xy):
    """Mean brightness in 1-pixel-wide rings around center_xy = (x, y)."""
    y, x = np.indices(img.shape)
    r = np.hypot(x - center_xy[0], y - center_xy[1]).astype(int)
    total = np.bincount(r.ravel(), weights=np.asarray(img, dtype=float).ravel())
    count = np.bincount(r.ravel())
    return total / np.maximum(count, 1)
