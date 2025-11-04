# Serial Module Reference

This guide summarises the serial-only implementation that lives under `src/QuantumAssemble.py` and `src/QAssemble/Serial`. Use it as a map when extending the solver, plumbing new data into the HDF5 workflow, or coordinating work between the Python and Fortran layers. The serial stack mirrors the MPI package, so preserving method signatures keeps both back ends aligned.

## Top-Level Driver (`src/QuantumAssemble.py`)
- **Run lifecycle**  
  The `Run` class bootstraps the calculation. Instantiating `Run(test=False)` calls `ReadInput`, stores the parsed control dictionary, and dispatches to `RunDiagE` for methods `tb`, `hf`, or `gw`. Passing `test=True` builds a lightweight `CorrelationFunction` instance without launching a full run.
- **Input discovery**  
  `ReadInput` executes `input.ini`, expecting `Crystal`, `Hamiltonian`, and `Control` sections. It writes a canonicalised copy to `<prefix>.h5` under `/input` (creating the file on first run) and compares subsequent executions to guard against stale prefixes.
- **Derived dictionaries**  
  The routine populates nested dictionaries for `crystal`, `ft`, `ham`, and `run`, filling defaults such as `Mode=FromScratch`. Helper `CheckKeyinString` raises early errors if required configuration keys are missing.
- **HDF5 bookkeeping**  
  When an existing `<prefix>.h5` is present, `ReadInput` mirrors the new settings against the stored `/input` group. It aborts if values diverge, forcing the user to pick a new prefix. Fresh runs dump the executed Python objects into the HDF5 tree through `Dict2Hdf5`.
- **Environment expectations**  
  The script requires the root directory in `$QAssemble` so that dynamic imports and native extensions resolve (`sys.path` adjustments are commented but preserved). Ensure dependencies such as `h5py`, `numpy`, and `mpi4py` are available; MPI-specific packages are no-ops in serial mode but must still import cleanly.

## Core Entry Points
- `src/QAssemble/Serial/CorrelationFunction.py`  
  `CorrelationFunction` orchestrates the full workflow. It parses the control dictionary produced in `src/QuantumAssemble.py` and exposes `TightBinding`, `HartreeFock`, and `GWApproximation`. Each solver builds crystal data (`Crystal`), frequency grids ( `DLR`), then calls into the fermionic/bosonic lattice helpers to compute self-energies, mix them, and persist results to `<prefix>.h5`.
- `src/QAssemble/Serial/Crystal.py`  
  Holds lattice metadata: lattice vectors, basis positions, spin count, and index maps between composite, fermionic, and bosonic spaces. Key helpers such as `FAtomOrb`, `BAtomOrb`, `OrbSpin2Composite`, and `Quad2Double` convert between different orbital labellings. Methods like `Kpath`, `KPoint`, and `MappingRVec` generate momentum grids that downstream modules reuse.


## Fermionic Lattice Modules
- `src/QAssemble/Serial/FLatDyn.py`  
  `FLatDyn` handles momentum-frequency (k, omega) Green's functions for fermions using a discrete Lehmann representation (`DLR`). Core methods include `F2T`/`T2F` transforms, `Moment` for high-frequency tails, and `K2R`/`R2K` for Fourier hopping between reciprocal and real space. Helper classes manage common workflows:  
  - `GreenBare` builds non-interacting propagators (`Cal`) and writes them to disk.  
  - `GreenInt` iteratively adjusts the chemical potential with `Occ`, `SearchMu`, and `UpdateMu`.  
  - `SigmaGWC` evaluates GW self-energies and computes renormalisation factors via `Zfactor`.  
  - `GreenAB` converts interpolated data between k-indexed layouts (`KI2KF`).
- `src/QAssemble/Serial/FLatStc.py`  
  Provides band-structure and static self-energy routines. `FLatStc.K2R`/`R2K` mirror the dynamic transforms, while `Band`, `DOS`, and `Visualization` furnish plotting utilities. Subcomponents:  
  - `NIHamiltonian` constructs k-space Hamiltonians and optional valley-resolved blocks.  
  - `SigmaHartree` and `SigmaFock` create mean-field corrections.  
  - `Hamiltonian` (`CalMu0`, `NumOfE`, `SearchMu`, `OccMixing`) controls charge self-consistency.  
  - `HamiltonianAB` offers k-path interpolation helpers for A/B sublattice presentations.
- `src/QAssemble/Serial/FPathDyn.py`  
  Evaluates dynamic quantities along high-symmetry paths. The constructor can rebuild `Crystal` and `DLR` from an HDF5 checkpoint when invoked with only a file name. `Inverse`, `R2K`, and `KArb` provide on-the-fly interpolation, while `MQEMWrapper` bridges to the Julia-based `MQEM` experiments if `MQEM.jl` is initialised.
- `src/QAssemble/Serial/FPathStc.py`  
  Static analogue of `FPathDyn`. It supports custom path transforms (`R2K`, `K2R`), slab projections (`Slab`, `SlabZmat`), and quick-look diagnostics like `Gaussian`, `Dos`, `Band`, and `FermiSurface`. `CheckGroup` is useful before reading optional HDF5 groups.

## Bosonic Lattice Modules
- `src/QAssemble/Serial/BLatDyn.py`  
  Controls bosonic lattice dynamics. `Inverse` builds composite orbital-spin blocks, `Moment`/`F2T`/`T2F` share the DLR pipeline with fermions, and `K2R`/`R2K` adapt the phase factors to bosonic indexing. High-level utilities `StcEmbedding`, `RT2mRmT`, and `TauF2TauB` assist with symmetry averaging, while `PolLat` and `WLat` compute polarisation and screened-interaction tensors, respectively.
- `src/QAssemble/Serial/BLatStc.py`  
  Provides static (`omega=0`) bosonic response support. In addition to the inverse and Fourier routines, `Mixing` stabilises the self-consistent loop, and `Dyson` wraps the shared Dyson solver. `VBare` evaluates bare Coulomb kernels (`Cal`, `LocPlusNonLoc`, `OhnoYukawa`) and writes parameter tables with `Save`.
- `src/QAssemble/Serial/BPathDyn.py` / `BPathStc.py`  
  Minimal wrappers that currently expose `R2K` transforms for k-path extraction from real-space datasets.



## Utility Layer
- `src/QAssemble/Serial/utility/DLR.py`  
  Wraps the discrete Lehmann representation grids. Constructor parameters set temperature, beta, and frequency cutoffs; `FT2F`, `FF2T`, `BT2F`, and `BF2T` perform high-accuracy imaginary-time/Matsubara conversions. The `TauDLR2Uniform` family exports non-uniform sampling onto uniform meshes when plotting.

## Working Tips
- Dynamic classes (`FLatDyn`, `BLatDyn`) expect a `DLR` instance seeded with the same beta/cutoff as the solver. When adding new routines, pass the `DLR` object rather than re-instantiating it.
- Real/reciprocal conversions apply phase factors derived from `Crystal.basisf`. Preserve these factors if you alter the basis ordering, otherwise the transforms will silently break hermiticity.
- Many methods previously called into the Fortran extension `QAFort`. The Python fallbacks in `utility/Fourier.py` and `utility/Dyson.py` are feature-complete but slower; profile large runs if you disable the compiled modules.
