# Static tensile solution

`run_fe.py` is the FE command entry. Its default path ends at M9 CAE construction, including save, close and protected-copy inspection. It submits no solver job unless `--solve-elastic` is explicitly supplied.

## Explicit solution controls

`--solve-elastic` requires `--load-axis` and `--strain`. The load axis is `x`, `y` or `z`; strain is dimensionless and must satisfy `0 < strain <= 0.05`. Percent strings, nonfinite values and nonpositive strains are rejected.

After successful construction, the original M10/M11 routines add the ideal tie and periodic constraints, apply the selected tensile loading and submit the job. The solution works from a protected copy of the constructed CAE and does not generate new geometry or remesh it. The CAE and resulting ODB remain in the output directory. A failed or incomplete job does not establish successful solution.

## Material and step definitions

The model uses `StaticStep` with `nlgeom=OFF`, Al elastic constants 69000 MPa and 0.33 with the Johnson-Cook plastic declaration, and SiC elastic constants 450000 MPa and 0.17. Coordinates are in micrometres and stress is in MPa. The full loading trajectory is not assumed to be a purely linear elastic perturbation.

The FE implementation, material definitions and runtime controls are in [run_fe.py](../run_fe.py). It prepares the required Abaqus scripts within the fresh output directory. The original CPU, memory-percentage, output-frequency, step and job-prefix defaults are retained; the memory percentage is a configuration ceiling, not a measured allocation.

## Outputs and failures

This entry retains CAE/ODB files and operational status records. It contains no ODB result extraction, curve export, modulus calculation or automatic summary generation.

If CAE construction fails, stop that request and generate another geometry with user-selected `--seed` and `--size-seed` values. Keep the new geometry and its acceptance record together and select a fresh FE output directory. The FE command does not change the seed, regenerate geometry, relax a tolerance or retry construction automatically.
