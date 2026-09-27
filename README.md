# Orbit Validation

Simulates the orbits of real satellites around Earth and Mars and measures how far a simple
two-body model drifts from NASA's official ephemerides (JPL Horizons).

Supported satellites: **ISS**, **Hubble** (Earth), **Mars Reconnaissance Orbiter** and
**Mars Odyssey** (Mars).

![ISS: simulated vs JPL Horizons](figures/iss_projections.png)

## How it works

1. **Reference data.** State vectors (position and velocity) are downloaded from the
   [JPL Horizons API](https://ssd.jpl.nasa.gov/horizons/) relative to the planet's centre,
   in the ICRF equatorial frame.
2. **Simulation.** The first Horizons state is used as the initial condition, and the
   two-body equations of motion

   $$\ddot{\mathbf r} = -\frac{\mu}{|\mathbf r|^3}\,\mathbf r$$

   are integrated with SciPy's adaptive Runge–Kutta method (DOP853), evaluated at exactly
   the same timestamps as the Horizons data.
3. **Validation.** The position error is computed at every epoch, summarised as RMSE,
   and plotted over time along with 3D and 2D views of both orbits.

## Results

Each satellite was simulated over 6 hours of Horizons data (1-minute steps, starting 2024-01-01).

| Satellite | Body  | Period (two-body) | RMSE (km) | Final error (km) |
|-----------|-------|-------------------|-----------|------------------|
| ISS       | Earth | 93.0 min          | 142.9     | 245.8            |
| Hubble    | Earth | 95.1 min          | 165.1     | 293.1            |
| MRO       | Mars  | 111.8 min         | 154.3     | 246.2            |
| Odyssey   | Mars  | 118.1 min         | 183.5     | 304.8            |

The computed orbital periods agree well with the published values for each spacecraft, which
confirms that the initial states and physical constants are correct. The position error, however,
grows steadily and reaches 250–300 km after six hours (roughly four orbits) for every satellite.

Interestingly, the Mars orbiters drift about as much as the Earth satellites even though Mars has
almost no atmosphere. This suggests that the main source of error is not drag but the planets'
oblateness (J2): the equatorial bulge changes the orbit in ways a point-mass model cannot capture,
and Mars' J2 is roughly twice as large as Earth's.

![ISS: position error over time](figures/iss_error.png)

The two-body model ignores the planet's oblateness (the J2 term), atmospheric drag and
third-body gravity, so some growth of the error over time is expected.

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/VasiLijeVook/orbit-validation.git
cd orbit-validation
pip install -r requirements.txt

# Download 6 hours of ISS data from Horizons, simulate, and save plots to figures/
python -m orbit_validation --satellite iss --fetch

# Re-run on the saved data without downloading again
python -m orbit_validation --satellite iss
```

Options: `--start` / `--stop` / `--step` set the Horizons time range (e.g. `--step 5m`),
`--show` opens the plots in a window, and `--help` lists everything.

## Tests

```bash
pytest
```

The tests check that the RMSE is computed correctly, that the integrator conserves orbital
energy, that a circular orbit returns to its starting point after one period, that Horizons
tables are parsed correctly, and that the full pipeline reproduces an exact two-body orbit
with near-zero error. They run automatically on every push via GitHub Actions.

## Project structure

```
orbit_validation/
├── bodies.py     # planet constants and Horizons IDs of the satellites
├── horizons.py   # downloading and parsing Horizons vector tables
├── dynamics.py   # equations of motion, integration, orbital energy and period
├── metrics.py    # position error and RMSE
├── plotting.py   # 3D orbit, 2D projections, error over time
└── cli.py        # command-line interface
tests/            # pytest test suite
```

## Next steps

- Add the J2 perturbation and check whether it removes most of the error, as the results above suggest.
- Implement fixed-step integrators (RK4, leapfrog) and compare their accuracy and energy
  conservation with the adaptive method.

## Background

Started as a university project in Jupyter notebooks and later refactored into a tested
Python package.
