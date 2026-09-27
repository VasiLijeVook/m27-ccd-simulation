import numpy as np

from ccdsim.compare import compare_images
from ccdsim.imaging import normalize_image
from ccdsim.model import m27_nebula_model, make_wcs, place_stars, simulate_ccd
from conftest import BACKGROUND, FWHM, M27, NEBULA_PEAK, NOISE, SCALE, SHAPE, ZERO_POINT


def simulate_with(wcs, scene):
    flux = 10 ** (-0.4 * (scene["mags"] - ZERO_POINT))
    stars = place_stars(SHAPE, wcs, scene["ra"], scene["dec"], flux)
    cx, cy = wcs.world_to_pixel(M27)
    nebula = m27_nebula_model(SHAPE, (cx, cy), angle_deg=30.0)
    image, _ = simulate_ccd(stars, nebula, NEBULA_PEAK, FWHM, BACKGROUND, NOISE, seed=99)
    return image


def test_simulation_with_correct_wcs_matches_the_observation(scene):
    sim = simulate_with(scene["wcs"], scene)
    metrics = compare_images(normalize_image(scene["image"]), normalize_image(sim), smooth_sigma=FWHM / 2.355)
    assert metrics["correlation"] > 0.8


def test_wrong_orientation_destroys_the_correlation(scene):
    # Regression test for the original bug: a north-up WCS used for a rotated image.
    sim = simulate_with(make_wcs(M27, SHAPE, SCALE, rotation_deg=0), scene)
    metrics = compare_images(normalize_image(scene["image"]), normalize_image(sim), smooth_sigma=FWHM / 2.355)
    assert metrics["correlation"] < 0.4
