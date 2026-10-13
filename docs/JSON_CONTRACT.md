# JSON files used by the README examples

The geometry commands write `geometry.json`. The mixed-aspect FE command in README uses [examples/vf30_mixed_ar12_20/geometry.json](../examples/vf30_mixed_ar12_20/geometry.json).

| Field | Content |
| --- | --- |
| `schema` | `pw85-ellipsoids-export`. |
| `periodic` | `true`. |
| `box` | Three box lengths; the examples use `[60, 60, 60]` micrometres. |
| `particles` | A list of particles with `id`, `a`, `b`, `c` and `p` fields. |

For each particle, `id` is an integer identifier. The positive scalars `a` and `b` are physical semiaxes in micrometres, with `a >= b`. The ellipsoid has semiaxes `(a, b, b)`. The three-component `c` field is the centre, and `p` is the three-component direction vector; `c` is not a scalar semiaxis.

Use the generated or packaged JSON without editing it. When using a generated file for FE construction, keep its companion `geometry_acceptance.json` in the same directory. Additional fields in generated files should be left in place.

See [GEOMETRY.md](GEOMETRY.md) for the two parameter sets and [GEOMETRY_TO_CAE.md](GEOMETRY_TO_CAE.md) for the FE input paths.
