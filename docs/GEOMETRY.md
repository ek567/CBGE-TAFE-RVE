# Geometry generation and acceptance

Run `python run_geometry.py` from the repository root. The public interface resolves the versioned [vf30-reference profile](../config/vf30-reference.json) and translates typed parameters to the bound geometry engine. `--show-config` prints that resolution without importing the scientific backend, generating particles or creating an output directory. Normal generation requires a fresh `--output-dir` outside the repository; existing output is preserved.

## Particle sizes and shapes

`--deq-truncnorm MU SIGMA MIN MAX` specifies the truncated normal distribution of equivalent spherical diameter, in micrometres. The native sampler estimates particle count from the third diameter moment, draws a population, and adjusts the count and re-samples until the requested volume-fraction band is met. It does not impose a fixed particle count or rescale a sampled population to force its volume fraction. A failed size admission is recorded separately from a geometry acceptance failure. See [`generate_ellipsoid_sizes_from_truncnorm`](../geometry/src/generation/size_generator_3d.py).

Each physical particle is an axisymmetric ellipsoid with semi-axes `(a, b, b)`. For equivalent diameter `d` and aspect ratio `rho = a/b`, the sampler uses `a = (d/2) rho^(2/3)` and `b = (d/2) rho^(-1/3)`, preserving the volume `pi d^3 / 6`.

`--aspect fixed:R` assigns the same ratio to every particle while their diameters still vary. `--aspect uniform:LO:HI` draws an independent ratio for each physical particle using a separate, reproducible size-seed-derived stream. It does not draw one ratio for an entire realization. See the [fixed and uniform branches](../geometry/src/generation/size_generator_3d.py).

The public minimum-gap factor uses the theoretical smallest semi-minor axis permitted by the declared diameter and ratio bounds:

```text
rho_max = R                       for fixed:R
rho_max = HI                      for uniform:LO:HI
b_theoretical_min = (MIN/2) rho_max^(-1/3)
delta_min = min_gap_factor * b_theoretical_min
```

This reference is determined before sampling; it is not the smallest semi-axis found in the sampled population. The native search adds `delta_min/2` to both stored semi-axes, then restores each particle's original physical sizes for final audits and JSON export. This is the implemented inflated-ellipsoid clearance constraint, rather than a separate certificate of exact Euclidean surface distance. See [public resolution](../public_cli/geometry_interface.py) and [`inflate_particles_min_distance_3d` / physical restoration](../geometry/src/constraints/min_distance_3d.py).

## GEO and orientation controls

GEO has distinct face, edge and corner safety-band widths and feasibility tolerances, all expressed in the input length unit. `--geo-width W` broadcasts one width; `--geo-widths FACE EDGE CORNER` sets the three independently. `--geo-tol T` and `--geo-tols FACE EDGE CORNER` behave similarly for tolerances. A scalar option and its three-value alternative are mutually exclusive. The native staged search activates GEO progressively, using the full requested widths and tolerances at the final stage. Final feasibility compares each maximum violation with its corresponding tolerance; a small GEO energy alone is not the acceptance criterion. See [native staged execution](../geometry/src/main.py) and [`_final_audit_verdicts`](../geometry/src/main.py).

Orientation targets describe the number-averaged second-order tensor `A = (1/N) sum(p_i p_i^T)` of normalized particle directions. The supported targets are:

| Public value | Target tensor |
| --- | --- |
| `1d-x` | `diag(1, 0, 0)` |
| `2d-xz` | `diag(1/2, 0, 1/2)` |
| `3d-isotropic` | `I/3` |
| `off` | ORI is disabled and is not a required final acceptance gate. |

Other axis or plane names are rejected because this engine does not provide those target modes. The [native target definitions](../geometry/src/model/orientation_targets_3d.py) and [tensor audit](../geometry/src/validation/orientation_isotropy_check_3d.py) determine these meanings.

Orientation acceptance uses the geometry engine's target-alignment audit and the resolved `orientation_tol`. For `3d-isotropic`, each eigenvalue of the final number-averaged tensor must differ from `1/3` by less than this tolerance. The public profile supplies `orientation_tol = 0.05`; the native isotropic mode without an explicit tolerance override defaults to `0.015`. The required target-alignment verdict is evaluated together with the physical PW, periodic and GEO feasibility checks. See [`check_orientation_isotropy`](../geometry/src/validation/orientation_isotropy_check_3d.py) and [`evaluate_public_export`](../public_cli/geometry_adapter.py).

## Budgets, retries and seeds

The engine enforces a shared ceiling of **7000 committed optimizer iterations per geometry candidate**, across all native stages. Native `--steps 100000` is the unchanged local invocation limit and does not increase that shared ceiling. The paper's 6000-iteration setting and this engine's 7000-iteration ceiling are different settings; a reproduction record must identify which was used. Line-search trials, diameter sampling, final audits and receipt writing are not counted as committed optimizer iterations. See [`RealizationIterationBudget`](../geometry/src/resource_budget.py), the [local parser default](../geometry/src/main.py), and the [shared stage-budget argument](../geometry/src/main.py).

`--realizations K` requests K independently accepted geometries. Each request may consume at most **21 geometry candidates**, using the bounded acceptance controller. Each candidate can use the initial size seed plus at most **50 additional size-seed replacements** for volume-fraction admission, so there may be up to 51 size-seed attempts. That size policy is separate from the 21 geometry-candidate policy. Exhaustion or a fatal diagnostic produces a failed terminal receipt, not a fabricated accepted geometry. All attempted candidates and seed lineage are retained. See [`plan_seed_pairs`](../geometry/geometry_acceptance_retry.py), [`run_request`](../geometry/geometry_acceptance_retry.py), and the [native size-retry parser contract](../geometry/src/main.py).

`--seed` controls position/orientation initialization and `--size-seed` controls the size/shape population. Both accept nonnegative integers. If `--seed` is omitted, the unchanged native default is used and recorded; it is not a recommended scientific seed. If `--size-seed` is omitted, it inherits the effective placement seed. For K greater than one, the original initial seed arithmetic is preserved: placement seed `base + 1000 + index`, size seed `size_base + index`, with a zero-based index. Subsequent candidates use the original collision-avoiding seed plan; receipts distinguish requested and effective size seeds. See [typed seed resolution](../public_cli/geometry_interface.py) and [native plan use](../public_cli/geometry_adapter.py).

Choose seed values for your study and retain them with the resolved configuration.

## Outputs

For each accepted realization the runner writes:

```text
OUTPUT_DIR/
  geometry_run_receipt.json
  realization_001/
    geometry.json
    geometry_acceptance.json
    native_retry_receipt.json
    attempts/a01/g.json
    attempts/a01/out.txt
    attempts/a01/err.txt
```

Additional realizations use `realization_002`, and additional candidates use `a02`, etc. The accepted `geometry.json` is a byte-preserving copy of the selected native JSON. Its particle fields `a` and `b` are physical semi-axes; `c` is the centre vector and `p` the direction vector, not a third scalar semi-axis.

`geometry_acceptance.json` binds the selected JSON SHA-256, source identity, resolved configuration, engine and final acceptance checks and actual seed lineage. Accepted single realizations use `GEOMETRY_ACCEPTED`; the aggregate receipt uses `GEOMETRY_BATCH_ACCEPTED` only when every requested realization is accepted. The independently checked physical PW and strict periodic audits, required GEO feasibility, and the required native target-alignment audit determine acceptance. A process exit code or native optimization `success` flag alone is insufficient. See [sidecar construction](../public_cli/geometry_adapter.py) and the [run status implementation](../public_cli/geometry_adapter.py).

Geometry generation does not automatically invoke Abaqus. For user-generated geometry, `run_fe.py` consumes the JSON together with its bound adjacent acceptance sidecar; keep both files when transferring a geometry. The packaged mixed-aspect geometry is a separate frozen input recognized through its exact file and source identity.
