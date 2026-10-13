# Parameters and implementation

The public interface exposes the parameters listed below. [The versioned profile](../config/vf30-reference.json) supplies omitted values. The implementation and final orientation acceptance rule are described in [GEOMETRY.md](GEOMETRY.md).

## Geometry

| Public control | Profile value | Native mapping and meaning |
| --- | --- | --- |
| `--profile` | `vf30-reference` | Loads the versioned profile and records its exact SHA-256. |
| `--box LX LY LZ` | `60 60 60` | `--box`; positive periodic box dimensions, micrometres. |
| `--vf` | `0.30` | `--vf-target`; a fraction strictly between 0 and 1. |
| `--vf-tol` | `0.005` | `--vf-abs-tol`; absolute volume-fraction admission tolerance. |
| `--deq-truncnorm MU SIGMA MIN MAX` | `12 2.4 7.2 16.8` | `--deq-mu`, `--deq-sigma`, `--deq-min`, `--deq-max`; equivalent-diameter distribution, micrometres, positive sigma and `MIN < MAX`. |
| `--aspect fixed:R` | `fixed:1.5` | `--aspect-ratio-mode fixed` and `--aspect-ratio-k`; every particle has the same ratio `R >= 1`. Diameters still vary. |
| `--aspect uniform:LO:HI` | Optional alternative | `--aspect-ratio-mode uniform`, `--aspect-ratio-min`, `--aspect-ratio-max`; independent ratio per particle, `1 <= LO < HI`. |
| `--min-gap-factor` | `0.1` | Resolves `--delta-min = F (MIN/2) rho_max^(-1/3)` from declared theoretical bounds. Native search inflates each semi-axis by half this value. |
| `--geo on/off` | `on` | Native `--geometry-distortion` / `--no-geometry-distortion`. On uses the original staged activation. |
| `--geo-width W` | Broadcast form | Broadcasts to native `--geo-d-face`, `--geo-d-edge`, `--geo-d-corner`; positive length. |
| `--geo-widths FACE EDGE CORNER` | `0.8 0.8 0.8` | Independently sets those three native safety-band widths, micrometres. |
| `--geo-tol T` | Broadcast form | Broadcasts three native per-kind violation tolerances; nonnegative length. |
| `--geo-tols FACE EDGE CORNER` | `0.01 0.01 0.01` | `--geo-face-violation-tol`, `--geo-edge-violation-tol`, `--geo-corner-violation-tol`; final maximum violations must be at most these values. |
| `--orientation` | `3d-isotropic` | `1d-x`, `2d-xz`, `3d-isotropic` bind the original target tensors. `off` disables native ORI and its required final gate. |
| `--orientation-tol` | `0.05` | Forwarded to the native target-alignment audit. This profile overrides the native isotropic default `0.015`. |
| `--overlap-tol` | `1e-8` | `--overlap-tol`; restored physical PW and strict periodic audit tolerance. Distinct from inflated search stopping tolerances. |
| `--realizations K` | `1` | Number of requested accepted geometries; each request uses the bounded candidate retry policy. |
| `--seed` | Native default if omitted | Position/orientation initialization seed; nonnegative integer, recorded as explicit or inherited native default. No recommended seed is implied. |
| `--size-seed` | Effective placement seed if omitted | Independent size/shape seed; nonnegative integer. Explicit values map to native `--size-seed`. |
| `--output-dir` | Required for generation | Fresh directory outside the repository; all output and retry artifacts are placed there. |
| `--show-config` | Off | Configuration-only path: no scientific backend import, calculation or output creation. |

The typed parser and native translation are in [`geometry_interface.py`](../public_cli/geometry_interface.py). The actual native argument definitions are in [`build_arg_parser`](../geometry/src/main.py). Sampling and shape formulas are in [`size_generator_3d.py`](../geometry/src/generation/size_generator_3d.py). Scalar and vector GEO controls are mutually exclusive within each pair.

## Retained internal controls and budget meaning

The public CLI does not expose an untyped native-argument escape hatch or optimizer tuning flags. Resolved metadata records the following bound controls:

| Control | Effective value | Source |
| --- | --- | --- |
| Shared committed iteration ceiling | 7000 per geometry candidate, across stages | [`resource_budget.py`](../geometry/src/resource_budget.py) |
| Local invocation maximum | Native `--steps 100000` | [`main.py`](../geometry/src/main.py) |
| Geometry candidates | At most 21 per requested accepted geometry | [`geometry_acceptance_retry.py`](../geometry/geometry_acceptance_retry.py) |
| Size-seed replacements | At most 50 extra attempts after the initial size seed | [`main.py`](../geometry/src/main.py) |
| Position/orientation update coefficients | Native `tau_c = tau_p = 1.5` | [`main.py`](../geometry/src/main.py) |
| Candidate pairs | `uniform_grid`, with extent filter and separation tolerance `1e-12` | [public translation](../public_cli/geometry_interface.py) |
| Inflated PW search tolerances | Stage 0: `0.002`; stages 1–3: `0.0001` | [`main.py`](../geometry/src/main.py) |
| Line search | Enabled; unchanged native parameter defaults | [native line-search defaults](../geometry/src/main.py) |

The paper's 6000-iteration limit is not this engine's 7000 committed-iteration ceiling. The native local `100000` limit also describes a different counter. Reproduction records retain the actual resolved budget and completed committed iterations. Final acceptance combines physical PW, periodic, required GEO feasibility and the native target-alignment audit.

## FE and optional tensile analysis

These controls configure FE construction and an explicitly requested solution from geometry without regenerating particles. `run_fe.py` is the command entry. Construction and solver completion are separate outcomes.

| Public control or fixed setting | Mapping |
| --- | --- |
| `run_fe.py --input-json` | Mandatory geometry input. User-generated inputs require an adjacent native accepted-geometry sidecar binding the exact JSON hash, source and final audits. The packaged mixed-aspect geometry is recognized by its exact file and source identity. |
| `--mesh-target-ratio` | Public default `0.06`, finite and positive; M6 target length is `ratio * max(box)`. The underlying function's default is `0.04`. The public adapter uses the existing M6 keyword parameter. |
| `--node-match-tol` | Optional positive coordinate-length tolerance mapped to native `tol_pair`; omission uses `1e-6 * max(box)`. Native `tol_plane` and ideal-tie position tolerance remain separately defined. |
| FE-only endpoint | Native M1–M9 construction, then explicit CAE save and close, followed by inventory inspection of a byte-verified copy and close. No M10/M11 or solver job from this endpoint. |
| Optional `--solve-elastic` | Explicitly enables the original M10/M11 solve after the M9 CAE exists. The result is an ODB; this option does not extract data, write a result CSV or calculate a modulus automatically. |
| `--load-axis` | `x`, `y` or `z` maps to the corresponding native axial loading case. |
| `--strain` | Positive dimensionless strain at most `0.05`; maps to native percentage as `100 * strain`. |
| Materials | Fixed Al matrix: `E = 69000 MPa`, `nu = 0.33`, Johnson–Cook plastic declaration. SiC: `E = 450000 MPa`, `nu = 0.17`. No public material override. |
| Interface and step | Ideal tie and `StaticStep` with `nlgeom = OFF` retained. The declared plastic law remains in the model; the complete path is not represented as a purely linear perturbation analysis. |

The FE parameter interface, runtime controls, material settings and numerical implementations are contained in [`run_fe.py`](../run_fe.py). The entry prepares the required Abaqus runtime scripts in the fresh output directory. It preserves the M9 save/close and protected-copy inspection checks; explicit tensile solution uses the original M10/M11 routines. The mesh ratio is adjustable, and mesh convergence must be assessed separately for the selected settings. No result extraction or curve/summary generation follows the solver.
