"""Building blocks of the simulated CCD image: WCS, PSF, stars, nebula, detector effects."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.signal import fftconvolve


def make_wcs(center, shape, pixel_scale_arcsec, rotation_deg=0.0):
    """Tangent-plane WCS centred on `center`, north up and east left, rotated by rotation_deg.

    Only used for tests and experiments - the real image uses the plate-solved WCS.
    """
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    s = pixel_scale_arcsec / 3600
    t = np.radians(rotation_deg)
    w.wcs.cd = np.array([[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]])
    return w


def sky_angle_to_pixel_angle(wcs, center, position_angle_deg):
    """Angle in the pixel grid (degrees from +x towards +y) of a direction on the sky
    given as a position angle (degrees from north through east)."""
    offset = center.directional_offset_by(position_angle_deg * u.deg, 1 * u.arcmin)
    x0, y0 = wcs.world_to_pixel(center)
    x1, y1 = wcs.world_to_pixel(offset)
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def gaussian_psf(fwhm_px):
    """Normalised Gaussian PSF kernel (atmospheric seeing + telescope optics)."""
    sigma = fwhm_px / 2.355
    size = max(int(np.ceil(8 * sigma)) | 1, 9)
    y, x = np.indices((size, size)) - size // 2
    psf = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return psf / psf.sum()


def place_stars(shape, wcs, ra_deg, dec_deg, flux_adu):
    """Image with each star's total flux in the pixel given by the WCS (before the PSF)."""
    image = np.zeros(shape)
    x, y = wcs.world_to_pixel(SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg))
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    np.add.at(image, (iy[inside], ix[inside]), np.asarray(flux_adu, dtype=float)[inside])
    return image


def aperture_flux(img, x, y, radius, sky_inner=None, sky_outer=None):
    """Aperture photometry: sum within `radius` of (x, y) minus the local sky.

    The sky level is the median of an annulus between sky_inner and sky_outer
    (default 2x and 3x the aperture radius). A local sky matters here because stars
    on top of the nebula would otherwise get the nebula's light added to them.
    """
    sky_inner = sky_inner or 2 * radius
    sky_outer = sky_outer or 3 * radius
    r = int(np.ceil(sky_outer))
    xi, yi = int(round(x)), int(round(y))
    cut = img[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(float)
    yy, xx = np.indices(cut.shape)
    dist = np.hypot(xx - (x - xi + r), yy - (y - yi + r))
    sky = np.median(cut[(dist >= sky_inner) & (dist <= sky_outer)])
    aperture = dist <= radius
    return float((cut[aperture] - sky).sum())


def calibrate_zero_point(img, wcs, ra_deg, dec_deg, mags, fwhm_px,
                         mag_range=(12.0, 16.0), saturation=60000.0):
    """Photometric zero point ZP so that flux_ADU = 10 ** (-0.4 * (mag - ZP)).

    Uses isolated, unsaturated catalogue stars measured with aperture photometry
    and takes the median, which is robust to the occasional bad measurement.
    Returns (zero_point, number_of_stars_used).
    """
    ny, nx = img.shape
    radius = 2.0 * fwhm_px
    margin = int(np.ceil(3 * radius)) + 5
    x, y = wcs.world_to_pixel(SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg))
    mags = np.asarray(mags, dtype=float)

    candidates = []
    for i in range(len(mags)):
        if not (mag_range[0] <= mags[i] <= mag_range[1]):
            continue
        if not (margin <= x[i] < nx - margin and margin <= y[i] < ny - margin):
            continue
        # Skip stars with a neighbour that is not much fainter nearby (it would
        # contaminate the aperture or the sky annulus).
        dist = np.hypot(x - x[i], y - y[i])
        close = (dist > 0) & (dist < 3 * radius) & (mags < mags[i] + 2.5)
        if close.any():
            continue
        yi, xi = int(round(y[i])), int(round(x[i]))
        if img[yi - 2:yi + 3, xi - 2:xi + 3].max() >= saturation:
            continue
        flux = aperture_flux(img, x[i], y[i], radius)
        if flux > 0:
            candidates.append(mags[i] + 2.5 * np.log10(flux))

    if len(candidates) < 5:
        raise ValueError(f"Only {len(candidates)} usable calibration stars - check the WCS and catalogue")
    return float(np.median(candidates)), len(candidates)


def m27_nebula_model(shape, center_xy, angle_deg):
    """Parametric dumbbell-shaped surface-brightness model of M27, peak normalised to 1.

    Two lobes, a central waist, a faint halo and a shell. The sizes (in pixels) are
    hand-chosen for a 0.39"/px image; angle_deg is the lobe axis in the pixel grid.
    """
    y, x = np.indices(shape)
    t = np.radians(angle_deg)
    dx, dy = x - center_xy[0], y - center_xy[1]
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)

    lobes = (np.exp(-((xr - 105) ** 2 / (2 * 95**2) + yr**2 / (2 * 155**2)))
             + np.exp(-((xr + 105) ** 2 / (2 * 95**2) + yr**2 / (2 * 155**2))))
    waist = 0.70 * np.exp(-(xr**2 / (2 * 55**2) + yr**2 / (2 * 235**2)))
    r_ell = np.sqrt((xr / 290) ** 2 + (yr / 380) ** 2)
    halo = 0.25 * np.exp(-(r_ell**2) / 2)
    shell = 0.35 * np.exp(-((r_ell - 0.70) ** 2) / (2 * 0.075**2))

    nebula = lobes + waist + halo + shell
    return nebula / nebula.max()


def calibrate_nebula_peak(img, nebula_model, background, core_level=0.6):
    """Nebula brightness (ADU above sky) at the model's peak, measured from the real image
    as the median excess in the model's bright core. The median ignores the stars."""
    core = nebula_model >= core_level
    excess = np.median(img[core]) - background
    mean_model = nebula_model[core].mean()
    return max(float(excess / mean_model), 0.0)


def simulate_ccd(star_image_adu, nebula_image, nebula_peak_adu, fwhm_px, background_adu,
                 background_noise_adu, gain_e_per_adu=1.0, hot_pixel_fraction=2e-5,
                 n_cosmic_rays=30, saturation_adu=65535.0, seed=42):
    """Turn the ideal scene into a CCD-like image.

    Steps: PSF blur -> sky background -> noise -> hot pixels and cosmic rays -> saturation.

    Noise: the background noise measured on the real image already contains both the
    sky photon noise and the read noise, so it is added once as a whole. Extra photon
    (Poisson) noise is added only for the light from the stars and the nebula.
    """
    rng = np.random.default_rng(seed)
    psf = gaussian_psf(fwhm_px)
    signal = (fftconvolve(star_image_adu, psf, mode="same")
              + nebula_peak_adu * fftconvolve(nebula_image, psf, mode="same"))
    signal = np.maximum(signal, 0)

    image = (signal + background_adu
             + rng.normal(0, background_noise_adu, signal.shape)
             + rng.normal(0, 1, signal.shape) * np.sqrt(signal / gain_e_per_adu))

    hot = rng.random(signal.shape) < hot_pixel_fraction
    image[hot] += rng.uniform(1000, 5000, hot.sum())

    ny, nx = signal.shape
    cy, cx = rng.integers(0, ny, n_cosmic_rays), rng.integers(0, nx, n_cosmic_rays)
    image[cy, cx] += rng.uniform(6000, 22000, n_cosmic_rays)

    return np.clip(image, 0, saturation_adu), signal
