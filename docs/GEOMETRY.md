# Geometry parameters for the README examples

Run the two geometry commands in [README.md](../README.md) from the repository root after installing the dependencies.

| Parameter | Example value | Use |
| --- | --- | --- |
| `--profile` | `vf30-reference` | Selects the settings supplied for these examples. |
| `--box` | `60 60 60` | Box lengths in micrometres. |
| `--vf` | `0.30` | Target particle volume fraction. |
| `--deq-truncnorm` | `12 2.4 7.2 16.8` | Equivalent-diameter mean, standard deviation, lower bound and upper bound, in micrometres. |
| `--aspect` | `fixed:1.5` | Fixed particle aspect ratio of 1.5. |
| `--aspect` | `uniform:1.2:2.0` | Particle aspect-ratio range from 1.2 to 2.0. |
| `--orientation` | `3d-isotropic` | Three-dimensional isotropic orientation target. |
| `--geo-width` | `0.8` | Face, edge and corner safety-band width in micrometres. |
| `--realizations` | `1` | Requests one geometry output. |
| `--output-dir` | The path shown in the command | A fresh output directory outside the repository. |

The fixed-aspect command uses `../runs/vf30_fixed_ar15`. The mixed-aspect command uses `../runs/vf30_mixed_ar12_20`. Create the `../runs` parent directory as shown in README; each named output directory must not already exist.

The geometry files used by the FE commands are:

```text
../runs/vf30_fixed_ar15/realization_001/geometry.json
../runs/vf30_mixed_ar12_20/realization_001/geometry.json
```

Keep each generated `geometry.json` together with its companion `geometry_acceptance.json` in the same directory. Use these files without editing them.

The packaged mixed-aspect file in `examples/vf30_mixed_ar12_20/geometry.json` is a separate input for the FE command shown in README.

Geometry generation does not start Abaqus automatically. If the geometry command stops without producing its output JSON, inspect the terminal message before starting FE construction. See [GEOMETRY_TO_CAE.md](GEOMETRY_TO_CAE.md) for the next command.
