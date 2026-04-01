# Quick Start

## 1. Prepare `input.ini`

QAssemble reads input from a file named `input.ini` written in plain Python dict syntax.
Below is a minimal example for a graphene-like two-site model with GW:

```python
Crystal = {
    'RVec': [[1,0,0],[0.5,0.866,0],[0,0,1]],
    'SOC': False,
    'CorF': 'F',                          # 'F' = Fractional, 'C' = Cartesian
    'Basis': [[[0.33333,0.33333,0],1],
              [[0.66667,0.66667,0],1]],
    'NSpin': 1,
    'NElec': 2,
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
            'Parameter': 'SlaterKanamori',
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

## 2. Run

=== "CLI (recommended)"

    ```bash
    qassemble
    ```

=== "Module execution"

    ```bash
    python -m QAssemble
    ```

=== "MPI parallel"

    ```bash
    mpirun -n <num_processors> qassemble
    mpirun -n <num_processors> python -m QAssemble
    ```


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
        ├── FPathStc.py         # Static fermionic path
        ├── FPathDyn.py         # Dynamic fermionic path
        ├── BLatStc.py          # Static bosonic lattice (bare/screened Coulomb)
        ├── BLatDyn.py          # Dynamic bosonic lattice (polarization, screened W)
        ├── BLocStc.py          # Static bosonic local site
        ├── BPathStc.py         # Static bosonic path
        ├── BPathDyn.py         # Dynamic bosonic path
        └── utility/
            ├── DLR.py          # Discrete Lehmann Representation transforms
            ├── Dyson.py        # Dyson equation solver
            ├── Fourier.py      # Lattice Fourier transforms
            ├── Common.py       # Shared utilities
            ├── Bare.py         # Bare Green's functions
            └── Mixing.py       # Mixing parameter control
```
