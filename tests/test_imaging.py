import numpy as np
import pytest

from ccdsim.imaging import detect_sources, estimate_background_noise, estimate_fwhm, normalize_image
from ccdsim.model import gaussian_psf


def test_one_saturated_pixel_does_not_crush_normalisation():
    # Regression test: dividing by the maximum made the whole real image almost black.
    rng = np.random.default_rng(0)
    img = rng.normal(1000, 10, (200, 200))
    img[100, 100] = 65535
    norm = normalize_image(img)
    assert np.median(norm) > 0.2


def test_background_and_noise_are_recovered():
    img = np.random.default_rng(0).normal(2380, 31, (600, 600))
    background, noise = estimate_background_noise(img)
    assert background == pytest.approx(2380, abs=1)
    assert noise == pytest.approx(31, rel=0.05)


def star_field(n_stars=30, n_hot=40, fwhm=6.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(1000, 10, (400, 400))
    psf = gaussian_psf(fwhm)
    h = psf.shape[0] // 2
    for _ in range(n_stars):
        x, y = rng.integers(30, 370, 2)
        img[y - h:y + h + 1, x - h:x + h + 1] += 2e5 * psf
    hx, hy = rng.integers(0, 400, n_hot), rng.integers(0, 400, n_hot)
    img[hy, hx] += 3000  # hot pixels: bright but only one pixel wide
    return img


def test_hot_pixels_are_not_counted_as_stars():
    # Regression test: hot pixels and cosmic rays used to inflate the source count.
    img = star_field(n_stars=30, n_hot=40)
    xs, _ = detect_sources(img, 1000, 10)
    assert 25 <= len(xs) <= 30


def test_fwhm_of_gaussian_stars_is_recovered():
    img = star_field(n_stars=30, n_hot=0, fwhm=6.0)
    xs, ys = detect_sources(img, 1000, 10)
    assert estimate_fwhm(img, xs, ys) == pytest.approx(6.0, rel=0.1)


def test_fwhm_is_not_inflated_by_noisy_background():
    # Regression test: second moments with negatives clipped to zero overestimated the FWHM.
    rng = np.random.default_rng(3)
    img = rng.normal(2380, 31, (400, 400))
    psf = gaussian_psf(6.0)
    h = psf.shape[0] // 2
    positions = rng.integers(30, 370, (40, 2))
    for x, y in positions:
        img[y - h:y + h + 1, x - h:x + h + 1] += 3e4 * psf
    assert estimate_fwhm(img, positions[:, 0], positions[:, 1]) == pytest.approx(6.0, rel=0.1)
