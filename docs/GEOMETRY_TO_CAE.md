# FE construction for the README examples

Use Abaqus 2024 with the launcher configured as shown in [README.md](../README.md) and [ENVIRONMENT.md](ENVIRONMENT.md). Run the commands from the repository root.

For the fixed-aspect example, use the generated JSON:

```powershell
python run_fe.py --input-json ../runs/vf30_fixed_ar15/realization_001/geometry.json --mesh-target-ratio 0.06 --output-dir ../runs/vf30_fixed_ar15_fe
```

For the mixed-aspect example, use the packaged JSON:

```powershell
python run_fe.py --input-json examples/vf30_mixed_ar12_20/geometry.json --mesh-target-ratio 0.06 --output-dir ../runs/vf30_mixed_ar12_20_fe
```

To use the generated mixed-aspect JSON instead, replace `--input-json` with `../runs/vf30_mixed_ar12_20/realization_001/geometry.json` and choose a fresh FE output directory. Keep the generated JSON and its companion `geometry_acceptance.json` together.

| Parameter | Use |
| --- | --- |
| `--input-json` | Geometry file to model. |
| `--mesh-target-ratio 0.06` | Sets the target mesh length to `0.06 * max(box)`; for a 60-micrometre box this is 3.6 micrometres. |
| `--output-dir` | A fresh output directory outside the repository. |

Successful construction writes `official_success/final.cae` inside the selected output directory. These example commands construct and save the FE model without submitting a solution job.

If construction fails, inspect the terminal message and Abaqus output before another request. A geometry JSON does not guarantee successful FE construction. Use a different geometry input and a fresh output directory when making another request; the FE command does not generate a replacement geometry automatically.
