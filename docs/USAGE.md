# QuantumAssemble Usage Guide

## 1. Overview

QuantumAssemble couples Python drivers with Fortran kernels to perform tight-binding (TB), Hartree–Fock (HF), and GW electronic structure calculations. The main entry point is `src/QuantumAssemble.py`, which reads a Python-style `input.ini`, stores all inputs and results in an HDF5 file, and dispatches either serial or MPI-enabled workflows under `src/QAssemble`.

## 2. Repository Layout

- `src/QuantumAssemble.py` – serial driver that parses `input.ini`, prepares an HDF5 checkpoint (`<Prefix>.h5`), and runs TB / HF / GW loops.
- `src/QAssemble/Serial` and `src/QAssemble/MPI` – domain logic for single-process and distributed executions. Mirror updates across both trees.
- `src/QAssemble/modules` – Fortran sources compiled into the `QAFort` Python extension (rebuilt with `make new`).
- `MQEM.jl` – Julia maximum-entropy package used for analytic continuation of spectral data.
- `arch.mk*` – compiler, library, and toolchain configuration templates. Copy or edit the version that matches your platform.

## 3. Prerequisites

1. **Python 3.9** with packages: `numpy`, `scipy`, `h5py`, `matplotlib`, `numba`, `finufft`, `mpi4py`, `mpi4py-fft`.
2. **Fortran toolchain**: `gfortran` (default) or Intel `ifort`, plus BLAS/LAPACK and FFTW development headers.
3. **Finite NUFFT library**: place a FINUFFT build under `${QAssemble}/finufft` or adjust `arch.mk` to point to your installation.
4. **MPI stack** (OpenMPI, MPICH, Intel MPI, etc.) for distributed runs.
5. **Julia ≥ 1.8** (optional) for `MQEM`.

Python dependencies can be installed with:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install numpy scipy h5py matplotlib numba finufft mpi4py mpi4py-fft
```

## 4. Environment Setup

1. Clone the repository and enter the project directory.
2. Export the project root so helper scripts can locate assets:
   ```bash
   export QAssemble=$(pwd)
   ```
3. Pick the matching `arch.mk` variant (`arch.mk`, `arch.mk_mac`, `arch.mk_intel`) and tailor the compiler, include paths, and library locations to your site.
4. Build the Fortran extension:
   ```bash
   make -C src/QAssemble/modules new
   ```
   This regenerates `.o`, `.mod`, and the `QAFort` shared object using the toolchain configured in `arch.mk`.
5. (Optional) Rebuild FINUFFT from the project root if you vend the library locally:
   ```bash
   make qa_finufft
   ```
   Edit the top-level `Makefile` or `arch.mk` if your FINUFFT lives elsewhere.

## 5. Writing `input.ini`

`input.ini` is executed with `exec`, so it must be valid Python code that defines three dictionaries: `Control`, `Crystal`, and `Hamiltonian`. Place the file next to the command you execute (`src/QuantumAssemble.py` reads `./input.ini`).

### 5.1 Control Block

```python
Control = {
    "Method": "hf",            # "tb", "hf", or "gw"
    "Prefix": "graphene_hf",   # produces graphene_hf.h5
    "Mode": "FromScratch",     # or "Restart" to resume from a saved HDF5 state
    "Mix": 0.1,                # linear mixing factor for SCF loops
    "NSCF": 200,               # maximum self-consistency iterations
    "ConstantW": 1.0,          # prefactor passed to correlation functions
    "MatsubaraCutOff": 200,
    "T": 300.0                 # specify either T (Kelvin) or beta (1/eV)
}
```

### 5.2 Crystal Block

```python
Crystal = {
    "RVec": [
        [2.46, 0.0, 0.0],
        [1.23, 2.13, 0.0],
        [0.0, 0.0, 15.0]
    ],
    "Basis": [
        [[0.0, 0.0, 0.0], 1],      # [fractional_position, orbitals_per_atom]
        [[1/3, 2/3, 0.0], 1]
    ],
    "CorF": "F",                   # "F"ractional (default) or "C"artesian input
    "NSpin": 2,
    "SOC": False,
    "NElec": 4.0,
    "KGrid": [12, 12, 1]
}
```

### 5.3 Hamiltonian Block

```python
Hamiltonian = {
    "OneBody": {
        "Hopping": {
            ((0, 0), (1, 0)): {
                -2.7: [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            },
            ((1, 0), (0, 0)): {
                -2.7: [[0, 0, 0]]
            }
        },
        "Onsite": {
            0: {(0, 0): 0.0, (1, 0): 0.0},   # spin channel -> {(atom, orb): energy}
            1: {(0, 0): 0.0, (1, 0): 0.0}
        },
        "Spin": False,
        "Site": False,
        "AntiSite": False,
        "Valley": False,
        "AntiValley": False,
        "AntiFerro": False
    },
    "TwoBody": {
        "Local": {
            "Parameter": "SlaterKanamori",  # "Slater", "Kanamori" also accepted
            "option": {
                (0, 0): {"l": 2, "U": 3.0, "J": 0.5},
                (1, 0): {"l": 2, "U": 3.0, "J": 0.5}
            }
        },
        "NonLocal": "None"  # or dicts enabling Ohno/JTH/OhnoYukawa interactions
    }
}
```

Key points:

- `Hopping` maps `(from_atom, from_orb), (to_atom, to_orb)` to a dictionary whose keys are complex amplitudes and values are lists of lattice translation vectors `[Rx, Ry, Rz]`.
- `Onsite` is organized by spin channel; each entry maps `(atom, orb)` tuples to onsite shifts.
- `TwoBody["Local"]` accepts `"SlaterKanamori"`, `"Slater"`, or `"Kanamori"`. Provide orbital angular momentum `l`, Coulomb `U`, exchange `J`, and, if needed, `Up` or explicit Slater integrals.
- `TwoBody["NonLocal"]` can be `"None"` or a structure with flags like `"Ohno": True`, `"JTH": True`, or `"OhnoYukawa": True`, plus parameter dictionaries (see `QuantumAssemble.py` for the expected fields).

On the first run, the driver writes the entire `Control`, `Crystal`, and `Hamiltonian` dictionaries into `<Prefix>.h5/input`. Subsequent runs verify that the file matches the current `input.ini`. If the inputs differ, change `Prefix` to avoid overwriting previous data.

## 6. Running the Serial Workflow

1. Ensure `input.ini` is present in your working directory.
2. Activate your Python environment and export `QAssemble` if not already set.
3. Launch the driver:
   ```bash
   python src/QuantumAssemble.py
   ```
4. The script prints progress for the chosen `Method` and writes results into `<Prefix>.h5`, grouped by calculation type (`tb`, `hf`, `gw`, …). Intermediate checkpoints (e.g., `hk.<iter>`, `sigh.<iter>`) are saved every 50 SCF steps.
5. To restart an HF run from an existing HDF5 checkpoint, set `"Mode": "Restart"` in `Control` and keep the same `Prefix`.

You can also instantiate the driver programmatically:

```python
from QuantumAssemble import Run
run = Run(test=True)  # loads control dictionaries without executing TB/HF/GW loops
```

## 7. MPI Workflows

Distributed kernels live under `src/QAssemble/MPI`. They rely on `mpi4py`, `mpi4py-fft`, and the `MPIManager` helper class.

To build a parallel driver:

```python
from mpi4py import MPI
from QAssemble.MPI.CorrelationFunction import CorrelationFunction

comm = MPI.COMM_WORLD
# Load control dicts the same way QuantumAssemble.py does
func = CorrelationFunction(cry=control["crystal"], ft=control["ft"], comm=comm)
```

Launch with `mpiexec -n <ranks> python your_driver.py`. Ensure `nprock * nprocf` matches the MPI world size when calling `MPIManager.Quary`.

The serial `QuantumAssemble.py` script can still be used alongside MPI-specific routines by importing `QAssemble.MPI` modules where needed.

## 8. Testing and Validation

- Export `QAssemble` so tests can locate compiled modules:
  ```bash
  export QAssemble=$(pwd)
  ```
- Spot-check Fourier transforms with:
  ```bash
  python src/QAssemble/MPI/utility/test/TestFourierFT.py
  ```
- Follow the existing `1e-6` absolute error tolerance when adding new regression tests under `src/QAssemble/*/utility/test/`.

## 9. Julia Maximum Entropy (`MQEM.jl`)

1. Instantiate dependencies:
   ```bash
   julia --project=MQEM.jl -e 'using Pkg; Pkg.instantiate()'
   ```
2. Run Julia scripts inside `MQEM.jl/src` to post-process spectral data exported from the HDF5 files.
3. Plotting helpers and sample gnuplot scripts live under `MQEM.jl/gnuplot_and_input`.

## 10. Troubleshooting

- **Compilation errors**: double-check `arch.mk` paths for FFTW, LAPACK, and FINUFFT libraries. Use absolute paths or environment variables to avoid hard-coded, host-specific directories.
- **Missing `QAFort`**: rebuild with `make -C src/QAssemble/modules new` after editing any Fortran sources.
- **MPI deadlocks**: ensure `MPIManager.Quary` receives consistent `(nk, nf, ntau, nprock, nprocf)` on every rank and that `nprock * nprocf` equals the total number of ranks.
- **Input mismatches**: if the driver aborts because `<Prefix>.h5` contains different inputs, rename the file or update `"Prefix"` in `Control`.
- **Environment detection**: scripts read `${QAssemble}` to locate compiled modules, so export it in every new shell session or add it to your shell profile.


