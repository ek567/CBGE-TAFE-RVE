# Dependencies and license boundaries

First-party software uses the MIT license. Dependencies retain their own licenses; the project license does not relicense their source, binaries or proprietary runtimes.

| Dependency | Role | Source / license |
| --- | --- | --- |
| NumPy | Geometry arrays and numerical operations | [NumPy](https://numpy.org/); package licenses and notices accompany the distribution |
| SciPy | Scientific numerical routines used by the geometry implementation | [SciPy](https://scipy.org/); package licenses and notices accompany the distribution |
| Matplotlib | Statistics and plots imported by the geometry closure | [Matplotlib](https://matplotlib.org/); its license agreement and bundled notices accompany the distribution |
| pw85 / pypw85 | Perram–Wertheim contact function | [sbrisard/pw85](https://github.com/sbrisard/pw85), BSD-3-Clause |
| pybind11 | Build-time Python/C++ binding headers | [pybind11](https://github.com/pybind/pybind11), BSD-3-Clause |
| Boost.Math | Build-time Brent minimization headers used by pw85 | [Boost](https://www.boost.org/), BSL-1.0 |
| setuptools and wheel | Building the external Python extension | Public Python package distributions with their own license notices |
| Abaqus | Commercial FE modelling and solution runtime | Separately obtained proprietary runtime and license |

The exact runtime pins are in `requirements-geometry.txt`; [INSTALLATION.md](INSTALLATION.md) gives the public source-build route. The external native extension is built from pw85 commit `515da5a74002c956c9e12b4918659a51b92eca37`, version 2.0, by Sébastien Brisard. Its [BSD-3-Clause license](https://github.com/sbrisard/pw85/blob/515da5a74002c956c9e12b4918659a51b92eca37/LICENSE) requires appropriate copyright, license-condition and disclaimer retention for redistribution and prohibits unapproved endorsement. The MIT selection for this repository does not remove those conditions.

Boost headers retain their original notices and [BSL-1.0 license](https://github.com/boostorg/boost/blob/boost-1.85.0/LICENSE_1_0.txt). pybind11 retains its [BSD-3-Clause license](https://github.com/pybind/pybind11/blob/v3.0.2/LICENSE). External dependency source is downloaded into a local build area rather than republished as part of this repository. Keep legally required notices with any redistributed dependency source or binaries.

The public repository contains first-party source and installation instructions. The repository does not distribute native extension binaries, wheels, virtual environments, compiler/SDK packages, proprietary runtimes, Abaqus installers, license files or CAE/ODB outputs. The original upstream source and package distributions remain the authority for their license terms.

The installation route uses public PyPI wheels and a pinned conda-forge Boost header archive. Local builds retain the dependencies' ownership and license terms. If dependency source or binaries are redistributed later, their actual distribution licenses and bundled notices must be reviewed and retained for that new scope.

See [ENVIRONMENT.md](ENVIRONMENT.md) for runtime requirements and [INSTALLATION.md](INSTALLATION.md) for the source-build route and compact-array interface check.

