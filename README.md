# CBGE-TAFE-RVE

## Environment and dependency installation

Use Windows x64 with a separate Python 3.11 environment. The geometry dependencies are NumPy 2.4.3, SciPy 1.17.1, Matplotlib 3.10.8 and the public pw85 2.0 distribution, imported as `pypw85`. Building pw85 requires Boost.Math 1.85.0 headers, an x64 C++17 compiler and the Windows SDK. The FE script was developed for Abaqus 2024. Abaqus 2024 must be installed and licensed separately for FE construction.

Follow [INSTALLATION.md](docs/INSTALLATION.md) to install the pinned Python packages and build pw85 from its public source. That guide includes the download checksums, compiler setup and installation commands. See [ENVIRONMENT.md](docs/ENVIRONMENT.md) for runtime requirements and [DEPENDENCIES.md](docs/DEPENDENCIES.md) for dependency licenses.

After installing, activate the environment and configure the launch process:

```powershell
.\.venv\Scripts\Activate.ps1
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CBGE_ABAQUS_COMMAND = (Get-Command abaqus -ErrorAction Stop).Source
New-Item -ItemType Directory -Force ../runs | Out-Null
```

Run the following commands from the repository root. Output directories must be fresh and outside the source tree; `../runs/` is a sibling output directory.

## Fixed aspect ratio 1.5

Box dimensions are 60 by 60 by 60 micrometres and target volume fraction is 0.30.

```powershell
python run_geometry.py --profile vf30-reference --box 60 60 60 --vf 0.30 --deq-truncnorm 12 2.4 7.2 16.8 --aspect fixed:1.5 --orientation 3d-isotropic --geo-width 0.8 --realizations 1 --output-dir ../runs/vf30_fixed_ar15
python run_fe.py --input-json ../runs/vf30_fixed_ar15/realization_001/geometry.json --mesh-target-ratio 0.06 --output-dir ../runs/vf30_fixed_ar15_fe
```

Keep the generated `geometry.json` and its adjacent `geometry_acceptance.json` together for FE input.

## Aspect ratio range [1.2, 2.0]

Box dimensions are 60 by 60 by 60 micrometres and target volume fraction is 0.30. Generate a geometry using:

```powershell
python run_geometry.py --profile vf30-reference --box 60 60 60 --vf 0.30 --deq-truncnorm 12 2.4 7.2 16.8 --aspect uniform:1.2:2.0 --orientation 3d-isotropic --geo-width 0.8 --realizations 1 --output-dir ../runs/vf30_mixed_ar12_20
```

The packaged mixed-aspect geometry can be used directly for FE construction:

```powershell
python run_fe.py --input-json examples/vf30_mixed_ar12_20/geometry.json --mesh-target-ratio 0.06 --output-dir ../runs/vf30_mixed_ar12_20_fe
```

To use the generated geometry from the command above, set `--input-json` to `../runs/vf30_mixed_ar12_20/realization_001/geometry.json` and keep its adjacent `geometry_acceptance.json`. See [GEOMETRY_TO_CAE.md](docs/GEOMETRY_TO_CAE.md) for FE input requirements.
