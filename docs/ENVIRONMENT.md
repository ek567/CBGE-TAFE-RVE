# Runtime environment

The installation instructions target Windows x64 and Python 3.11. Use a separate virtual environment and an x64 C++ build shell for the `pw85` extension.

| Component | Installation version or requirement |
| --- | --- |
| Python | 3.11, x64 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.8 |
| pw85 distribution / pypw85 binding | 2.0 |
| Upstream pw85 commit | `515da5a74002c956c9e12b4918659a51b92eca37` |
| pybind11 | 3.0.2 |
| setuptools / wheel / pip | 79.0.1 / 0.45.1 / 24.0 |
| Boost.Math headers | 1.85.0 |
| C++ compiler | MSVC with C++17 and the Windows SDK |
| Source and execution character set | UTF-8 |

Set numerical thread caps in the PowerShell process that launches geometry generation:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Use the virtual environment's interpreter. Output directories must be outside the source tree; keep dependency downloads, builds and temporary files out of version control.

Abaqus is installed and licensed separately. Geometry runs in the Python environment above; FE commands dispatch the installed Abaqus launcher. The mesh target length is `mesh_target_ratio × max(box_x, box_y, box_z)`, with a routine ratio of `0.06`. Mesh scale and GEO width are independent parameters.

Bind the installed launcher before running an FE command:

```powershell
$env:CBGE_ABAQUS_COMMAND = (Get-Command abaqus -ErrorAction Stop).Source
```

If the installation exposes a differently named launcher, resolve that installed command instead. The advanced FE command's `--abaqus-command` option supplies a launcher for one invocation and takes precedence over `CBGE_ABAQUS_COMMAND`. Use `--help` to inspect the option.

The optional solver uses the runtime settings defined in [run_fe.py](../run_fe.py). Its memory percentage is a configuration ceiling, not a fixed allocation or measured model demand. FE construction does not automatically enable the solver. Other operating systems and Python/compiler combinations may need different dependency build instructions.
