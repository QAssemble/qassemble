# QAssemble

**QAssemble** is a pure-Python quantum simulation package for calculating electronic properties of materials using free energy functional approaches. Built entirely with the Python standard library and a minimal set of well-established scientific packages — no compiled extensions or domain-specific frameworks required.

## Why Pure Python?

QAssemble is intentionally implemented in **pure Python**, meaning:

- No C/C++/Fortran extensions beyond what NumPy/SciPy already provide
- No proprietary or hard-to-install domain-specific libraries
- Readable, hackable source code — every algorithm is visible and modifiable
- Easy to install, easy to extend, and easy to understand

## Features

- **Methods**:
  - Tight-Binding (TB) — non-interacting band structure
  - Hartree-Fock (HF) — mean-field theory (restricted/unrestricted)
  - GW Approximation (GW) — many-body perturbation theory
- **Advanced Numerics**:
  - Discrete Lehmann Representation (DLR) for high-precision imaginary-time / Matsubara frequency transforms
  - Dyson equation solver for renormalized Green's functions
  - k-space / real-space Fourier transforms with phase-correct basis handling
  - High-frequency tail fitting for asymptotic accuracy
- **Coulomb Interactions**:
  - Local: Slater-Kanamori, Slater, Kanamori parameterizations
  - Non-local: Ohno, Ohno-Yukawa, J-threading (JTH)
- **Parallelization**: MPI-parallelized implementations via `mpi4py` (with graceful serial fallback)
- **Input/Output**: `.ini`-based configuration and HDF5 data storage via `h5py`
- **Crystal Structure**: Lattice vectors, basis positions, k-point grids, spin-orbit coupling (SOC)

## Dependencies

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

## Installation

### From source (recommended)

```bash
git clone https://github.com/Mo-Seong-Jun/QAssemble.git
cd QAssemble
pip install .
```

### Editable install (for development)

```bash
pip install -e .
```

After installation, the `qassemble` command will be available in your terminal.

## Usage

### 1. Prepare Input

Create an `input.ini` file in your working directory. It defines the crystal structure, Hamiltonian parameters, and run settings:

```python
Crystal = {
    'RVec': [[1,0,0],[0.5,0.866,0],[0,0,1]],
    'SOC': False,
    'CorF': 'F',                          # 'F' = Fractional, 'C' = Cartesian
    'Basis': [[[0.33333,0.33333,0],1],
              [[0.66667,0.66667,0],1]],
    'NSpin': 1,
    'NElec': 2,                           # Total number of electrons per spin
    'KGrid': [25,25,1]
}

Hamiltonian = {
    'OneBody': {
        'Hopping': {
            ((0,0),(1,0)): {
                1.0: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        },
        'Onsite': {
            0: {(0,0): 0.0, (1,0): 0.0}
        }
    },
    'TwoBody': {
        'Local': {
            'Parameter': 'SlaterKanamori',  # 'Slater', 'Kanamori', or 'SlaterKanamori'
            'option': {
                (0,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0},
                (1,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0}
            }
        },
        'NonLocal': {
            ((0,0),(1,0)): {
                0.20: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        }
    }
}

Control = {
    'Method': 'gw',           # 'tb', 'hf', or 'gw'
    'Prefix': 'my_calc',
    'NSCF': 20000,
    'Mix': 0.1,
    'T': 2000,
    'MatsubaraCutOff': 100,
    'ConstantW': 1.0
}
```

### 2. Run Simulation

Using the installed CLI command:

```bash
qassemble
```

Using Python module execution:

```bash
python -m QAssemble
```

<!-- Parallel execution with MPI:

```bash
mpirun -n <num_processors> qassemble
mpirun -n <num_processors> python -m QAssemble
``` -->

## Directory Structure

```
QAssemble/
├── pyproject.toml              # Package configuration and dependencies
├── README.md
└── src/
    ├── QAssemble.py            # Legacy entry point (backward compatible)
    └── QAssemble/
        ├── __init__.py         # Package exports and version
        ├── __main__.py         # python -m QAssemble support
        ├── cli.py              # CLI entry point (qassemble command)
        ├── run.py              # Run class (input parsing and execution)
        ├── Crystal.py          # Lattice geometry, k-point grids, index mappings
        ├── CorrelationFunction.py  # Top-level workflow coordinator (TB / HF / GW)
        ├── FLatStc.py          # Static fermionic lattice (Hamiltonian, HF self-energy)
        ├── FLatDyn.py          # Dynamic fermionic lattice (Green's functions via DLR)
        ├── FLocStc.py          # Static fermionic local site
        ├── FLocDyn.py          # Dynamic fermionic local site
        ├── FPathStc.py         # Static fermionic path
        ├── FPathDyn.py         # Dynamic fermionic path
        ├── BLatStc.py          # Static bosonic lattice (bare/screened Coulomb)
        ├── BLatDyn.py          # Dynamic bosonic lattice (polarization, screened W)
        ├── BLocStc.py          # Static bosonic local site
        ├── BLocDyn.py          # Dynamic bosonic local site
        ├── BPathStc.py         # Static bosonic path
        ├── BPathDyn.py         # Dynamic bosonic path
        ├── Projector.py        # Projection utilities
        └── utility/
            ├── DLR.py          # Discrete Lehmann Representation transforms
            ├── Dyson.py        # Dyson equation solver
            ├── Fourier.py      # Lattice Fourier transforms
            ├── Common.py       # Shared utilities
            ├── Bare.py         # Bare Green's functions
            └── Mixing.py       # Mixing parameter control
```

### Module Naming Convention

| Prefix | Meaning |
|---|---|
| `F` | Fermionic |
| `B` | Bosonic  |
| `Lat` | Lattice  |
| `Loc` | Local  |
| `Path` | Path  |
| `Stc` | Static |
| `Dyn` | Dynamic |

## Configuration Reference

| Section | Key | Description |
|---|---|---|
| `Control` | `Method` | Calculation type: `"tb"`, `"hf"`, `"gw"` |
| `Control` | `Mode` | `"FromScratch"` or `"Restart"` |
| `Control` | `Prefix` | Output HDF5 filename prefix |
| `Control` | `NSCF` | Max SCF iterations |
| `Control` | `Mix` | Mixing parameter for self-consistency |
| `Control` | `T` | Temperature in Kelvin |
| `Control` | `MatsubaraCutOff` | Matsubara frequency cutoff |
| `Control` | `ConstantW` | Constant W parameter for GW |
| `Crystal` | `RVec` | 3x3 lattice vectors |
| `Crystal` | `Basis` | Basis atom positions and orbital counts |
| `Crystal` | `KGrid` | k-point grid `[Nx, Ny, Nz]` |
| `Crystal` | `NElec` | Number of electrons per spin |
| `Crystal` | `SOC` | Enable spin-orbit coupling |
| `Crystal` | `NSpin` | Number of spin channels |
| `Crystal` | `CorF` | Coordinate type: `"F"` (fractional) or `"C"` (Cartesian) |
| `Hamiltonian` | `Hopping` | One-body hopping terms |
| `Hamiltonian` | `Onsite` | On-site energy terms |
| `Hamiltonian` | `Parameter` | Coulomb parameterization: `"SlaterKanamori"`, `"Slater"`, `"Kanamori"` |
