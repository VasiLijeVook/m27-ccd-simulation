"""Star catalogue from Gaia DR3.

The stars are fetched from VizieR (a copy of Gaia DR3 hosted in Strasbourg, usually fast and
reliable). If that fails, the official Gaia archive is tried instead. The result is saved to a
local file, so the download only happens once.
"""

from pathlib import Path

from astropy import units as u
from astropy.table import Table

VIZIER_GAIA_DR3 = "I/355/gaiadr3"


def _from_vizier(center, radius_arcmin, mag_limit, timeout):
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"],
                    column_filters={"Gmag": f"<{mag_limit}"}, row_limit=-1, timeout=timeout)
    result = vizier.query_region(center, radius=radius_arcmin * u.arcmin, catalog=VIZIER_GAIA_DR3)
    if len(result) == 0:
        raise RuntimeError("VizieR returned no stars")
    t = result[0]
    return Table({"ra": t["RA_ICRS"], "dec": t["DE_ICRS"], "phot_g_mean_mag": t["Gmag"]})


def _from_gaia_archive(center, radius_arcmin, mag_limit):
    from astroquery.gaia import Gaia

    query = f"""
    SELECT ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {center.ra.deg}, {center.dec.deg}, {radius_arcmin / 60.0})) = 1
      AND phot_g_mean_mag < {mag_limit}
    """
    t = Gaia.launch_job_async(query).get_results()
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])
    return Table({"ra": t["ra"], "dec": t["dec"], "phot_g_mean_mag": t["phot_g_mean_mag"]})


def query_gaia(center, radius_arcmin=10.0, mag_limit=19.0, cache_path=None, timeout=120):
    """Gaia DR3 stars (ra, dec, phot_g_mean_mag) within radius_arcmin of center.

    If cache_path exists, the stars are read from it; otherwise they are downloaded
    (VizieR first, then the Gaia archive) and saved there. Needs an internet connection
    the first time. There is deliberately no synthetic fallback: random stars cannot be
    compared with a real image.
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"Using saved catalogue {cache_path}")
        return Table.read(cache_path)

    try:
        print("Downloading Gaia DR3 stars from VizieR...")
        table = _from_vizier(center, radius_arcmin, mag_limit, timeout)
    except Exception as error:
        print(f"VizieR failed ({error}); trying the Gaia archive, which can be slow...")
        table = _from_gaia_archive(center, radius_arcmin, mag_limit)

    table = table[[not bool(m) for m in getattr(table["phot_g_mean_mag"], "mask", [False] * len(table))]]
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.write(cache_path, overwrite=True)
        print(f"Saved catalogue to {cache_path}")
    return table
