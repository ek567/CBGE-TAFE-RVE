# Installation

These instructions target 64-bit Python 3.11 on Windows with a separate virtual environment, public PyPI packages and a local build of the public `pw85` source. Abaqus is installed separately.

Use an x64 Developer PowerShell with the Visual Studio C++ build tools and Windows SDK available. `cl.exe` must be discoverable in that shell. The upstream extension requires C++17. A standard PowerShell without the developer compiler environment may not find the compiler.

Start in the repository root. These commands create project-local environments, downloads and build files; keep them outside commits. No existing environment is needed.

```powershell
python --version
Get-Command cl.exe
$projectRoot = (Get-Location).Path
New-Item -ItemType Directory -Path ".deps/downloads", ".deps/sources", ".deps/temp", ".deps/wheels" -Force | Out-Null
$env:TEMP = Join-Path $projectRoot ".deps/temp"
$env:TMP = $env:TEMP
python -m venv --copies .venv
$pythonExe = Join-Path $projectRoot ".venv/Scripts/python.exe"
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple pip==24.0 setuptools==79.0.1 wheel==0.45.1 pybind11==3.0.2
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-geometry.txt
```

The requirements file pins the geometry dependencies and transitive runtime packages. `pypw85` is the import name of the separately built `pw85` distribution. An unrelated package or an assumed wheel under the import name is not a substitute.

Download the fixed upstream source and the pinned Boost header package:

```powershell
Invoke-WebRequest "https://codeload.github.com/sbrisard/pw85/zip/515da5a74002c956c9e12b4918659a51b92eca37" -OutFile ".deps/downloads/pw85-515da5a74002c956c9e12b4918659a51b92eca37.zip"
Invoke-WebRequest "https://conda.anaconda.org/conda-forge/win-64/libboost-headers-1.85.0-h57928b3_4.conda" -OutFile ".deps/downloads/libboost-headers-1.85.0-h57928b3_4.conda"
if ((Get-FileHash ".deps/downloads/pw85-515da5a74002c956c9e12b4918659a51b92eca37.zip" -Algorithm SHA256).Hash.ToLowerInvariant() -ne "fff8c9f7b7c98bd1bdacec09d01e778bd666de9a64d28b88d5b6749926326b3c") { throw "pw85 source archive checksum mismatch." }
if ((Get-FileHash ".deps/downloads/libboost-headers-1.85.0-h57928b3_4.conda" -Algorithm SHA256).Hash.ToLowerInvariant() -ne "c1b5f878403534ac1b2d1b84d5c6e522f5b988ff3904ce246f1272b6b3d85f54") { throw "Boost header package checksum mismatch." }
Expand-Archive ".deps/downloads/pw85-515da5a74002c956c9e12b4918659a51b92eca37.zip" ".deps/sources/pw85"
& $pythonExe -c "import zipfile; zipfile.ZipFile('.deps/downloads/libboost-headers-1.85.0-h57928b3_4.conda').extractall('.deps/boost-archive')"
New-Item -ItemType Directory -Path ".deps/boost", ".deps/boost-info" -Force | Out-Null
$boostPayload = Get-ChildItem ".deps/boost-archive" -Filter "pkg-*.tar.zst" -File
$boostInfo = Get-ChildItem ".deps/boost-archive" -Filter "info-*.tar.zst" -File
tar.exe -xf $boostPayload.FullName -C ".deps/boost"
tar.exe -xf $boostInfo.FullName -C ".deps/boost-info"
```

These checksums identify the pinned download archives. The upstream Git commit identifies the source revision. If a download checksum differs, verify that the source/package identity is correct before continuing.

Build without modifying the upstream source. `/utf-8` is required for the original non-ASCII documentation strings when compiling with MSVC; it changes the source/execution character set, not the numerical algorithm. Preserve the upstream license files and source notices.

```powershell
$boostInclude = (Resolve-Path ".deps/boost/Library/include").Path
$wheelDir = Join-Path $projectRoot ".deps/wheels"
$env:CL = "$env:CL /utf-8"
Push-Location ".deps/sources/pw85/pw85-515da5a74002c956c9e12b4918659a51b92eca37/python"
& $pythonExe -m pip --version
& $pythonExe -B setup.py build_ext --force --include-dirs $boostInclude
& $pythonExe -m pip --isolated wheel --no-build-isolation --no-deps --no-cache-dir --wheel-dir $wheelDir .
Pop-Location
& $pythonExe -m pip --isolated install --no-index --no-deps ".deps/wheels/pw85-2.0-cp311-cp311-win_amd64.whl"
& $pythonExe -m pip --isolated check
```

Perform a tiny dependency API check before running a geometry case:

```powershell
$env:MPLCONFIGDIR = Join-Path $projectRoot ".deps/matplotlib-config"
& $pythonExe -I -B -c "import numpy as n, scipy, matplotlib, pypw85; q=n.array([1.,0.,0.,1.,0.,1.]); out=n.empty(2); rc=pypw85.contact_function(n.array([3.,0.,0.]),q,q,out); assert rc==0 and n.allclose(out,[2.25,0.5]); print(pypw85.__file__); print(out)"
```

The native binding expects float64 displacement arrays with three entries, compact symmetric matrices `[xx, xy, xz, yy, yz, zz]` with six entries, and a mutable two-entry output array `[contact_function, lambda]`. A 3×3 matrix passed directly is not that compact representation.

The analytic sphere check confirms that the installed extension accepts the required compact-array interface. It does not generate an RVE or run FE analysis.

For later geometry commands, use this environment's interpreter and the documented thread caps:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
& $pythonExe run_geometry.py --help
& $pythonExe run_fe.py --help
```

Abaqus is a separate licensed runtime. Geometry runs in the Python environment above; FE drivers dispatch the explicitly supplied Abaqus launcher. Abaqus executables, licenses, CAE/ODB files and installers are not included. Configure that launcher through the public FE command rather than using a local-path default.

For other platforms and compilers, consult the [upstream installation documentation](https://github.com/sbrisard/pw85/blob/515da5a74002c956c9e12b4918659a51b92eca37/docs/installation.html).

