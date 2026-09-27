"""Metrics comparing a real and a simulated image on the same pixel grid."""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity


def compare_images(real_norm, sim_norm, smooth_sigma=0.0, signal_mask=None):
    """Compare two normalised (0-1) images.

    Pixel noise is independent in the two images, so in a mostly-empty sky it dominates
    every pixel-by-pixel metric. Smoothing both images by roughly the PSF size
    (smooth_sigma, in pixels) suppresses the noise while keeping stars and nebula.
    If signal_mask is given, the correlation is also computed only on pixels that
    contain light from stars or the nebula.
    """
    if smooth_sigma > 0:
        real_norm = gaussian_filter(real_norm, smooth_sigma)
        sim_norm = gaussian_filter(sim_norm, smooth_sigma)
    residual = real_norm - sim_norm
    metrics = {
        "ssim": float(structural_similarity(real_norm, sim_norm, data_range=1)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "correlation": float(np.corrcoef(real_norm.ravel(), sim_norm.ravel())[0, 1]),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
    }
    if signal_mask is not None:
        metrics["correlation_signal"] = float(
            np.corrcoef(real_norm[signal_mask], sim_norm[signal_mask])[0, 1])
    return metrics
