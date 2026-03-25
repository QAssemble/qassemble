# Serial Module Reference

This guide summarises the serial implementation that lives under `src/QuantumAssemble.py` and `src/QAssemble`. Use it as a map when extending the solver, plumbing new data into the HDF5 workflow, or coordinating work between the Python and Fortran layers.

## Top-Level Driver (`src/QuantumAssemble.py`)
- **Run lifecycle**
  The `Run` class bootstraps the calculation. Instantiating `Run(test=False)` calls `ReadInput`, stores the parsed control dictionary, and dispatches to `RunDiagE` for methods `tb`, `hf`, or `gw`. Passing `test=True` builds a lightweight `CorrelationFunction` instance without launching a full run.
- **Input discovery**
  `ReadInput` executes `input.ini`, expecting `Crystal`, `Hamiltonian`, and `Control` sections. It writes a canonicalised copy to `<prefix>.h5` under `/input` (creating the file on first run) and compares subsequent executions to guard against stale prefixes.
- **Derived dictionaries**
  The routine populates nested dictionaries for `crystal`, `ft`, `ham`, and `run`, filling defaults such as `Mode=FromScratch`. Helper `CheckKeyinString` raises early errors if required configuration keys are missing.
- **HDF5 bookkeeping**
  When an existing `<prefix>.h5` is present, `ReadInput` mirrors the new settings against the stored `/input` group. It aborts if values diverge, forcing the user to pick a new prefix. Fresh runs dump the executed Python objects into the HDF5 tree through `Dict2Hdf5`. Additional helpers `Hdf52Dict`, `CheckInput`, `ChangeInput`, and `CompareDict` manage round-trip serialisation and input validation.
- **Environment expectations**
  The script requires the root directory in `$QAssemble` so that dynamic imports and native extensions resolve. Ensure dependencies such as `h5py`, `numpy`, and `mpi4py` are available; MPI-specific packages are no-ops in serial mode but must still import cleanly.

## Core Entry Points
- `src/QAssemble/CorrelationFunction.py`
  `CorrelationFunction` orchestrates the full workflow. It parses the control dictionary produced in `src/QuantumAssemble.py` and exposes `TightBinding`, `HartreeFock`, and `GWApproximation`. Each solver builds crystal data (`Crystal`), frequency grids (`DLR`), then calls into the fermionic/bosonic lattice helpers to compute self-energies, mix them, and persist results to `<prefix>.h5`. The helper `SCFCheck` evaluates self-consistency convergence between iterations.
- `src/QAssemble/Crystal.py`
  Holds lattice metadata: lattice vectors, basis positions, spin count, and index maps between composite, fermionic, and bosonic spaces. Key helpers such as `FAtomOrb`, `BAtomOrb`, `OrbSpin2Composite`, `Composite2OrbSpin`, and `Quad2Double` convert between different orbital labellings. Methods like `Kpath`, `KPoint`, and `MappingRVec` generate momentum grids that downstream modules reuse. Additional methods include `Boson2Fermion`, `Boson2Full`, `SetFullBasis`, `Projector` for projection setup, and symmetry helpers `R2mRMapping`, `R2mR`, `RT2mRmT`, `T2mT`.

## Frequency / Temperature Grids
- `src/QAssemble/FTGrid.py`
  `FTGrid` generates uniform Matsubara and imaginary-time grids from temperature (or beta) and an energy cutoff. Properties `Omega`, `Nu`, and `Tau` provide fermionic Matsubara frequencies, bosonic Matsubara frequencies, and imaginary-time points respectively. Used as a fallback when `DLR` grids are not required.


## Fermionic Lattice Modules
- `src/QAssemble/FLatDyn.py`
  `FLatDyn` handles momentum-frequency (k, omega) Green's functions for fermions using a discrete Lehmann representation (`DLR`). Core methods include `F2T`/`T2F` transforms, `Moment` for high-frequency tails, and `K2R`/`R2K` for Fourier hopping between reciprocal and real space. Additional methods provide `Inverse`, `Mixing`, `Dyson`, `ChemEmbedding`, `StcEmbedding`, `Spectral`, `R2KArb`, `KArb`, `Diagonalize`, and symmetry operations `R2mR`/`T2mT`/`TauB2TauF`. Helper classes manage common workflows:
  - `GreenBare` builds non-interacting propagators (`Cal`) and writes them to disk (`Save`).
  - `GreenInt` iteratively adjusts the chemical potential with `CalMu0`, `Occ` (property), `SearchMu`, `UpdateMu`, and `NumOfE`.
  - `SigmaGWC` evaluates GW self-energies (`Cal`), computes the static part (`SigmaStc`), and renormalisation factors via `Zfactor`.
  - `GreenAB` converts interpolated data between k-indexed layouts (`KI2KF`).
- `src/QAssemble/FLatStc.py`
  Provides band-structure and static self-energy routines. `FLatStc` exposes `Inverse`, `K2R`/`R2K`, `Band`, `DOS`, `Visualization`, `Mixing`, `Dyson`, `ChemEmbedding`, `R2KArb`, `Diagonalize`, `HermitianCheck`, `SortKpoint`, and `KValley`. Subcomponents:
  - `NIHamiltonian` constructs k-space Hamiltonians (`Cal`, `Save`) and optional valley-resolved blocks (`Valley`, `AntiValley`).
  - `SigmaHartree` and `SigmaFock` create mean-field corrections (`Cal`, `Save`).
  - `Hamiltonian` controls charge self-consistency: `CalMu0`, `NumOfE`, `SearchMu`, `Occ` (property), `UpdateMu`, `OccMixing`, `Save`.
  - `HamiltonianAB` offers k-path interpolation helpers for A/B sublattice presentations (`KI2KF`).
  - `ZFactor` computes quasiparticle renormalisation from the frequency-dependent self-energy (`Cal`, `Save`).
  - `SigmaStc` extracts the static component of the GW self-energy (`Cal`, `Save`).
- `src/QAssemble/FPathDyn.py`
  Evaluates dynamic quantities along high-symmetry paths. The constructor can rebuild `Crystal` and `DLR` from an HDF5 checkpoint when invoked with only a file name. `Inverse`, `R2K`, and `KArb` provide on-the-fly interpolation. `MQEMWrapper` and `MQEMPrepare` bridge to the Julia-based `MQEM` experiments. `Spectral` computes spectral functions along paths.
- `src/QAssemble/FPathStc.py`
  Static analogue of `FPathDyn`. It supports custom path transforms (`R2K`, `K2R`), slab projections (`Slab`, `SlabZmat`, `SlabKpoint`), and diagnostics like `Gaussian`, `Dos`, `Band`, `FermiSurface`, `Occ`, and `Moments`. `Reshape` rearranges k-point data layouts.

## Bosonic Lattice Modules
- `src/QAssemble/BLatDyn.py`
  Controls bosonic lattice dynamics. `Inverse` builds composite orbital-spin blocks, `Moment`/`F2T`/`T2F` share the DLR pipeline with fermions, and `K2R`/`R2K` adapt the phase factors to bosonic indexing. Additional methods include `Mixing`, `Dyson`, `StcEmbedding`, `RT2mRmT`, `TauF2TauB`, `R2KArb`, `Save`, and index-space conversions (`Quad2Double`, `Double2Quad`, `Double2Full`, `Full2Double`, `Quad2Full`, `Full2Quad`). High-level classes:
  - `PolLat` computes polarisation tensors (`Cal`, `Save`).
  - `WLat` computes screened-interaction tensors (`Cal`, `Save`).
- `src/QAssemble/BLatStc.py`
  Provides static (`omega=0`) bosonic response support. Core methods: `Inverse`, `K2R`/`R2K`, `Mixing`, `Dyson`, `Save`, `R2KArb`, `HermitianCheck`, plus the same index-space conversions as `BLatDyn`. Subcomponent:
  - `VBare` evaluates bare Coulomb kernels (`Cal`, `LocPlusNonLoc`, `OhnoYukawa`, `OhnoParameter`, `JTHPotential`) and writes parameter tables with `Save`.
- `src/QAssemble/BPathDyn.py` / `BPathStc.py`
  Minimal wrappers that expose `R2K` transforms for k-path extraction from real-space datasets.

## Local Bosonic Module
- `src/QAssemble/BLocStc.py`
  `BLocStc` manages static local (impurity-level) bosonic quantities. Methods include `Inverse`, `Mixing`, `Dyson`, `Save`, `Imp2Loc`/`Loc2Imp` for mapping between impurity and local representations, `Arr2Dict`/`Dict2Arr` for converting between array and dictionary formats, and index-space conversions. Subcomponent:
  - `VLoc` handles local interaction vertices: `SetLocalInteracting`, `GenOnsite`, `SlaterKanamori`, `SlaterParameter`, `KanamoriParameter`, `AngularIntegral`, `RotationMatrix`, `Spherical2Cubic`, and `GetUijklComCTQMC`.

## Utility Layer
- `src/QAssemble/utility/DLR.py`
  Wraps the discrete Lehmann representation grids. Constructor parameters set temperature, beta, and frequency cutoffs; `FT2F`, `FF2T`, `BT2F`, and `BF2T` perform high-accuracy imaginary-time/Matsubara conversions. The `TauDLR2Uniform` family (including `TauDLR2Points`, `TauDLR2Uniform_v2`) exports non-uniform sampling onto uniform meshes when plotting. Additional methods: `TauUniform`, `MatsubaraFermionUniform`, `MatsubaraBosonUniform`, `TauUniform2DLR`, `MatsubaraDLR2Uniform`, `T2mT`, `TauF2TauB`, `TauB2TauF`.
- `src/QAssemble/utility/Fourier.py`
  Static methods for Fourier transforms across all lattice types. Moment extraction: `FLocDynM`, `FLatDynM`, `BLocDynM`, `BLatDynM`. Real/reciprocal conversions: `FLatStcK2R`/`R2K`, `FLatDynK2R`/`R2K`, `BLatStcK2R`/`R2K`, `BLatDynK2R`/`R2K`, `FPathStcR2K`, `FPathDynR2K`.
- `src/QAssemble/utility/Dyson.py`
  Static Dyson-equation solvers for every lattice/local combination: `FLocStc`, `FLatStc`, `FLocDyn`, `FLatDyn`, `BLocStc`, `BLocDyn`, `BLatStc`, `BLatDyn`.
- `src/QAssemble/utility/Bare.py`
  Static methods for bare propagator construction. Scalar versions: `FFreq`, `FTau`, `BFreq`, `BTau`. Matrix versions for local and lattice: `FLocFreq`, `FLatFreq`, `FLocTau`, `FLatTau`, `BLocFreq`, `BLatFreq`, `BLocTau`, `BLatTau`.
- `src/QAssemble/utility/Common.py`
  General numerical helpers: `MatInv` (matrix inversion), `HermitianEigenCmplx` (Hermitian diagonalisation), `SplineCmplx` / `FderivCmplx` (complex spline interpolation and derivatives), `BernoulliPolynomial`, `EulerPolynomial`, `FactorialInt`, `Ttind` (Chebyshev-node index mapping for tau grids), `Gcoeff` (high-frequency expansion coefficients), and `MinDistance`.

## Stub / Placeholder Modules
The following modules exist but contain only placeholder or fully commented-out code:
- `src/QAssemble/utility/Embedding.py` — empty `Embedding` class for future bath-embedding support.
- `src/QAssemble/utility/Projection.py` — empty `Projection` class for future basis-projection support.
- `src/QAssemble/utility/Mixing.py` — empty `Mixing` class.
- `src/QAssemble/Projector.py` — fully commented-out MPI projector code.
- `src/QAssemble/FLocStc.py`, `FLocDyn.py`, `BLocDyn.py` — entirely commented out.

## Working Tips
- Dynamic classes (`FLatDyn`, `BLatDyn`) expect a `DLR` instance seeded with the same beta/cutoff as the solver. When adding new routines, pass the `DLR` object rather than re-instantiating it.
- Real/reciprocal conversions apply phase factors derived from `Crystal.basisf`. Preserve these factors if you alter the basis ordering, otherwise the transforms will silently break hermiticity.
- Many methods previously called into the Fortran extension `QAFort`. The Python fallbacks in `utility/Fourier.py` and `utility/Dyson.py` are feature-complete but slower; profile large runs if you disable the compiled modules.
