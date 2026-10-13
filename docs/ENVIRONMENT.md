# Environment for the README examples

Use Windows x64, a separate Python 3.11 environment for geometry generation, and a separately installed and licensed Abaqus 2024 for FE construction.

| Component | Version or requirement |
| --- | --- |
| Python | 3.11, x64 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.8 |
| pw85 distribution / pypw85 binding | 2.0 |
| pybind11 | 3.0.2 |
| setuptools / wheel / pip | 79.0.1 / 0.45.1 / 24.0 |
| Boost.Math headers | 1.85.0 |
| C++ compiler | MSVC with C++17 and the Windows SDK |
| Compiler character set | UTF-8 |
| Abaqus | 2024 |

Follow [INSTALLATION.md](INSTALLATION.md) before running the examples. Activate the Python environment and set the four numerical thread caps in the same PowerShell session:

```powershell
.\.venv\Scripts\Activate.ps1
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CBGE_ABAQUS_COMMAND = (Get-Command abaqus -ErrorAction Stop).Source
New-Item -ItemType Directory -Force ../runs | Out-Null
```

Run the README commands from the repository root. The `../runs` parent directory holds the example outputs; each geometry or FE `--output-dir` must be a fresh directory that does not already exist.

See [DEPENDENCIES.md](DEPENDENCIES.md) for dependency licenses.
