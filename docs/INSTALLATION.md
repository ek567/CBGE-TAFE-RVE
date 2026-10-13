# Installation

Install the geometry dependencies for the README examples using 64-bit Python 3.11 on Windows, a separate virtual environment and the public `pw85` source. Abaqus 2024 must be installed and licensed separately.

Use an x64 Developer PowerShell with the Visual Studio C++ build tools and Windows SDK available. `cl.exe` must be discoverable in that shell. The upstream extension requires C++17. A standard PowerShell without the developer compiler environment may not find the compiler.

Start in the repository root. These commands create the Python environment and dependency build directories.

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

Build pw85 with the MSVC UTF-8 character set enabled. Retain the upstream license files and source notices.

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

Check that the installed packages can be imported:

```powershell
$env:MPLCONFIGDIR = Join-Path $projectRoot ".deps/matplotlib-config"
& $pythonExe -I -B -c "import numpy, scipy, matplotlib, pypw85; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```

Use this environment's interpreter and the thread caps below for the README commands:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
& $pythonExe run_geometry.py --help
& $pythonExe run_fe.py --help
```

The FE commands use Abaqus 2024. Set the installed launcher in the same PowerShell session:

```powershell
$env:CBGE_ABAQUS_COMMAND = (Get-Command abaqus -ErrorAction Stop).Source
New-Item -ItemType Directory -Force ../runs | Out-Null
```

Run the geometry and FE commands in [README.md](../README.md) from the repository root.

