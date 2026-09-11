# Installation

## Install from PyPI

QAssemble is published on [PyPI](https://pypi.org/project/QAssemble/):

```bash
pip install QAssemble
```

This is the recommended way to install QAssemble. There is no compilation
step and no compiler is required — QAssemble is pure Python. After
installation, the `qassemble` command is available in your terminal.

## Requirements

QAssemble requires Python 3.9 or later. `pip` installs the following
packages automatically:

| Package | Purpose |
|---|---|
| [NumPy](https://numpy.org/) | Array operations and linear algebra |
| [SciPy](https://scipy.org/) | Eigensolvers, interpolation, special functions |
| [h5py](https://www.h5py.org/) | HDF5-based data storage |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [pydlr](https://github.com/flatironinstitute/libdlr) | Discrete Lehmann Representation |
| [SymPy](https://www.sympy.org/) | Wigner 3j symbols and Gaunt coefficients |
| [pymatgen](https://pymatgen.org/) | Crystal structure utilities |

## Install from source

To work on QAssemble itself, clone the repository and install it in editable
mode with the test dependencies:

```bash
git clone https://github.com/QAssemble/qassemble.git
cd qassemble
pip install -e ".[test]"
```

The `test` extra adds [pytest](https://docs.pytest.org/) and
[nbmake](https://github.com/treebeardtech/nbmake), which are used to run the
test suite and to execute the example notebooks.
