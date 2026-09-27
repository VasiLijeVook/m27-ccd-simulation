import numpy as np
import pytest
from astropy import units as u

from ccdsim.model import (calibrate_nebula_peak, calibrate_zero_point, gaussian_psf,
                          make_wcs, place_stars, sky_angle_to_pixel_angle)
from conftest import BACKGROUND, FWHM, M27, NEBULA_PEAK, SCALE, SHAPE, ZERO_POINT


def test_psf_is_normalised():
    assert gaussian_psf(6.0).sum() == pytest.approx(1.0)


def test_north_points_down_in_an_image_rotated_by_180_degrees():
    wcs = make_wcs(M27, SHAPE, SCALE, rotation_deg=180)
    north = M27.directional_offset_by(0 * u.deg, 1 * u.arcmin)
    _, y_center = wcs.world_to_pixel(M27)
    _, y_north = wcs.world_to_pixel(north)
    assert y_north < y_center


def test_sky_angle_follows_image_rotation():
    upright = sky_angle_to_pixel_angle(make_wcs(M27, SHAPE, SCALE, 0), M27, 0)
    rotated = sky_angle_to_pixel_angle(make_wcs(M27, SHAPE, SCALE, 175.5), M27, 0)
    assert upright == pytest.approx(90, abs=0.1)  # north is +y
    turned = abs((rotated - upright + 180) % 360 - 180)  # direction of rotation doesn't matter
    assert turned == pytest.approx(175.5, abs=0.1)


def test_place_stars_conserves_flux():
    wcs = make_wcs(M27, SHAPE, SCALE)
    image = place_stars(SHAPE, wcs, [M27.ra.deg] * 3, [M27.dec.deg] * 3, [1.0, 2.0, 3.0])
    assert image.sum() == pytest.approx(6.0)


def test_zero_point_is_recovered(scene):
    zp, n_used = calibrate_zero_point(scene["image"], scene["wcs"], scene["ra"], scene["dec"],
                                      scene["mags"], FWHM)
    assert n_used >= 10
    assert zp == pytest.approx(ZERO_POINT, abs=0.1)


def test_nebula_brightness_is_recovered(scene):
    peak = calibrate_nebula_peak(scene["image"], scene["nebula"], BACKGROUND)
    assert peak == pytest.approx(NEBULA_PEAK, rel=0.2)
