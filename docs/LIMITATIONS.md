# Scope and limitations

The example input templates use Box=60 μm, Vf=0.30, fixed aspect ratio 1.5 or independent uniform aspect ratios over [1.2,2.0], and mesh target ratio 0.06. A template defines input settings; neither a template nor the packaged example geometry establishes statistical or mesh convergence. Changing the mesh ratio alone does not reproduce the manuscript's convergence study.

Orientation targets are limited to off, 1d-x, 2d-xz and 3d-isotropic. Other axis or plane names are rejected. When orientation control is enabled, acceptance follows the geometry engine's target-alignment audit at the configured tolerance.

The minimum-gap factor uses the theoretical minimum short semiaxis from the declared diameter and aspect bounds. Search inflates each semiaxis by half the resolved delta. This is an inflated-ellipsoid constraint, rather than an exact Euclidean surface-clearance calculation. GEO safety widths and residual tolerances are separate controls.

Geometry generation can exhaust its candidate or iteration budget. Example settings do not guarantee an accepted geometry or a completed CAE; completion must be established from the individual run's terminal status and required output checks. CAE construction and solver completion are separate stages. Static tensile loading includes the declared material law and must not be interpreted as a purely linear perturbation over the entire trajectory.

Accepted geometry can still fail native topology or periodic correspondence checks during CAE construction; the tool stops and retains diagnostics. To try a different initialization, explicitly generate and accept new geometry with a different seed and use a fresh FE output directory; mesh and tolerance settings are not changed automatically.

The original static tensile and periodic-constraint functions are available through `run_fe.py`. Solution requires explicit `--solve-elastic` options and successful CAE construction. Check the actual solver completion status before using its ODB. The FE command retains CAE/ODB files and does not include result extraction, curve export or summary generation.

FE execution requires a separately licensed Abaqus installation. MIT licensing of this software does not replace external dependency or Abaqus license terms.
