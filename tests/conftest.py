import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from ccdsim.model import m27_nebula_model, make_wcs, place_stars, simulate_ccd

M27 = SkyCoord(ra=299.90155 * u.deg, dec=22.7212 * u.deg)
SHAPE = (800, 800)
SCALE = 0.39
ZERO_POINT = 25.0
FWHM = 6.0
BACKGROUND, NOISE = 2380.0, 31.0
NEBULA_PEAK = 150.0


@pytest.fixture(scope="session")
def scene():
    """A synthetic 'real' observation: random catalogue stars and the nebula model,
    imaged with a WCS rotated by 175.5 deg like the Vidojevica image."""
    rng = np.random.default_rng(1)
    wcs = make_wcs(M27, SHAPE, SCALE, rotation_deg=175.5)
    n = 400
    sky = wcs.pixel_to_world(rng.uniform(20, SHAPE[1] - 20, n), rng.uniform(20, SHAPE[0] - 20, n))
    mags = rng.uniform(11.5, 18.5, n)
    flux = 10 ** (-0.4 * (mags - ZERO_POINT))

    cx, cy = wcs.world_to_pixel(M27)
    nebula = m27_nebula_model(SHAPE, (cx, cy), angle_deg=30.0)
    stars = place_stars(SHAPE, wcs, sky.ra.deg, sky.dec.deg, flux)
    image, _ = simulate_ccd(stars, nebula, NEBULA_PEAK, FWHM, BACKGROUND, NOISE, seed=7)
    return {"wcs": wcs, "image": image, "ra": sky.ra.deg, "dec": sky.dec.deg,
            "mags": mags, "nebula": nebula}
