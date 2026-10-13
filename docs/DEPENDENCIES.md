# Dependencies for the README examples

The first-party software uses the MIT license. Each dependency retains its own license.

| Dependency | Use | Source / license |
| --- | --- | --- |
| NumPy | Geometry arrays and numerical operations | [NumPy](https://numpy.org/); distribution license and notices. |
| SciPy | Numerical routines | [SciPy](https://scipy.org/); distribution license and notices. |
| Matplotlib | Plotting support | [Matplotlib](https://matplotlib.org/); distribution license and notices. |
| pw85 / pypw85 | Ellipsoid contact function | [sbrisard/pw85](https://github.com/sbrisard/pw85); BSD-3-Clause. |
| pybind11 | Python/C++ build headers | [pybind11](https://github.com/pybind/pybind11); BSD-3-Clause. |
| Boost.Math | C++ build headers | [Boost](https://www.boost.org/); BSL-1.0. |
| setuptools and wheel | Building the Python extension | License notices in their package distributions. |
| Abaqus 2024 | FE construction | Separately installed proprietary runtime and license. |

Install the versions in `requirements-geometry.txt` and build pw85 using [INSTALLATION.md](INSTALLATION.md).

Retain the upstream notices and license files supplied with pw85, pybind11 and Boost. Their license terms are provided in the [pw85 license](https://github.com/sbrisard/pw85/blob/515da5a74002c956c9e12b4918659a51b92eca37/LICENSE), [pybind11 license](https://github.com/pybind/pybind11/blob/v3.0.2/LICENSE), and [Boost license](https://github.com/boostorg/boost/blob/boost-1.85.0/LICENSE_1_0.txt). The project MIT license does not replace those terms or the Abaqus license.
