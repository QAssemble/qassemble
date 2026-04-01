# Installation

## Requirements

QAssemble requires Python 3.9 or later and the following packages:

| Package | Purpose |
|---|---|
| [NumPy](https://numpy.org/) | Array operations and linear algebra |
| [SciPy](https://scipy.org/) | Eigensolvers, interpolation, special functions |
| [h5py](https://www.h5py.org/) | HDF5-based data storage |
| [mpi4py](https://mpi4py.readthedocs.io/) | MPI parallelization |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [pydlr](https://github.com/flatironinstitute/libdlr) | Discrete Lehmann Representation |
| [SymPy](https://www.sympy.org/) | Wigner 3j symbols and Gaunt coefficients |
| [pymatgen](https://pymatgen.org/) | Crystal structure utilities |


## Install from GitHub

### From source (recommended)

Clone the repository and install:

```bash
git clone https://github.com/QAssemble/qassemble.git
cd QAssemble
pip install .
```

### Editable install (for development)

```bash
git clone https://github.com/QAssemble/qassemble.git
cd QAssemble
pip install -e .
```

After installation, the `qassemble` command will be available in your terminal.

No compilation step is required — QAssemble is pure Python.
