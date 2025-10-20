# QuantumAssemble_DEMO

QuantumAssemble couples Python drivers with Fortran kernels to perform tight-binding, Hartree–Fock, and GW calculations on crystalline systems. The project ships both serial and MPI execution paths alongside a Julia-based maximum-entropy post-processing toolkit.

## Quick Start

1. Create and activate a Python 3.9 environment, then install the required packages:
   ```bash
   pip install numpy scipy h5py matplotlib numba finufft mpi4py mpi4py-fft
   ```
2. Export the project root and rebuild the Fortran extension:
   ```bash
   export QAssemble=$(pwd)
   make -C src/QAssemble/modules new
   ```
3. Prepare an `input.ini` describing the `Control`, `Crystal`, and `Hamiltonian` dictionaries, then launch:
   ```bash
   python src/QuantumAssemble.py
   ```

See [`docs/USAGE.md`](docs/USAGE.md) for detailed setup instructions, configuration examples, and testing guidance.
