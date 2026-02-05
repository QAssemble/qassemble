# QAssemble

**QAssemble** is a Python-based quantum simulation package for calculating electronic properties of materials using diagrammatic expansion methods. It supports parallel execution via MPI and provides solvers for Tight-Binding (TB), Hartree-Fock (HF), and GW Approximation (GW).

## Features

- **Methods**:
  - Tight-Binding (TB)
  - Hartree-Fock (HF) (Restricted/Unrestricted)
  - GW Approximation (GW)
- **Parallelization**: Efficient parallel computing support using `mpi4py`.
- **Input/Output**: Flexible input configuration via `.ini` files and data storage using HDF5.
- **Crystal Structure**: Support for defining complex optimization of lattice vectors, basis positions, and orbital configurations.
- **Hamiltonian**:
  - One-body terms: Hopping, Onsite energy, Spin-Orbit Coupling (SOC).
  - Two-body terms: Local and Non-local Coulomb interactions (Slater-Kanamori, Ohno, etc.).

## Dependencies

- Python >= 3.9
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [h5py](https://www.h5py.org/)
- [mpi4py](https://mpi4py.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/) (for plotting)

## Installation

Clone the repository:

```bash
git clone https://github.com/sangkookchoi/DiagE.git
cd DiagE
```

Ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install numpy scipy h5py mpi4py matplotlib
```

## Usage

1. **Prepare Input**: Create an `input.ini` file in your working directory. This file should define the crystal structure, Hamiltonian parameters, and run control settings.

2. **Run Simulation**:

   For serial execution:
   ```bash
   python3 src/QuantumAssemble.py
   ```

   For parallel execution (using MPI):
   ```bash
   mpirun -n <num_processors> python3 src/QuantumAssemble.py
   ```

## Directory Structure

- `src/QAssemble`: Core source code for Quantum Assembly modules.
  - `MPI`: MPI-parallelized implementations.
  - `Serial`: Serial implementations.
  - `utility`: Helper utilities (FFT, DLR, etc.).
- `src/QuantumAssemble.py`: Main execution script.

## Configuration (`input.ini`)

The `input.ini` file supports sections for:
- `[Control]`: Run method (TB, HF, GW), mode (FromScratch, Restart), and parameters (mixing, NSCF steps).
- `[Crystal]`: Lattice definition, basis, K-grid, etc.
- `[Hamiltonian]`: Hopping, onsite energies, and Coulomb interaction parameters.

<!-- ## License

[MIT License](LICENSE) (or appropriate license) -->
