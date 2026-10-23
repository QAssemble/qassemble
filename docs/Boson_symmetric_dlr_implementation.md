# Common Lambda and K+ boson DLR implementation

Implemented on 2026-10-23, starting from `0294e8f` on
`DLR_GW_EDMFT_nocausal`.

## Behavior

- Both statistics use `Lambda = beta * cutoff`. `TauUniform` is unchanged.
- The global chemical-potential bracket uses `[-cutoff, cutoff]` plus the
  existing Hartree shift/spread, independently of Matsubara node locations.
- `Control['BosonSymmetry']` defaults to `True`. `False` selects the native
  pydlr boson basis, while retaining the new common bandwidth.
- K+ poles and independent time/frequency nodes use deterministic pivoted QR.
  The nonnegative pole grid includes zero. The zero-mode kernels are `1` in
  time and `beta * delta(nu, 0)` in frequency.
- Exposed time and Matsubara nodes include all mirror partners. Their counts
  can differ from one another and from the coefficient rank. Coefficient
  recovery averages mirror partners before a square LU solve.
- Odd and imaginary components outside the real/even sector produce a
  `QAssemble` warning when their relative maximum exceeds
  `max(100 * eps, 1e-10)`. Uniform fitting retains all signed input rows and
  minimizes the data residual with real coefficients.
- `T2mT` uses a node permutation, preserving non-even fermion data.

## Plan adjustments

- The uniform-fit identity matrix and coefficient-return shape now use the
  coefficient rank rather than the exposed frequency-node count.
- Existing correction-specific and causal-QP tests explicitly use
  `symmetry=False`; general uniform-fit tests cover both bases.
- Direct `CausalBosonProjector` use with K+ raises `NotImplementedError`.
  Its single-pole moment normalization is incompatible with the pair kernel.
  The active local no-causal path returns before constructing this projector.
- BLatDyn/BLocDyn already distinguish time/frequency dimensions, and the
  fermionic consumers of W already preserve batch axes. No edits were needed
  in those consumers; new tests exercise their transforms.

## Local verification

Command: `PYTHONPATH=$PWD/src python -m pytest src/QAssemble/utility/test/ -q`.
The final run set `MPLCONFIGDIR=/private/tmp/qassemble-mpl` for a writable cache.

| Stage | Passed | Failed |
|---|---:|---:|
| Original baseline | 445 | 22 |
| Common Lambda only | 440 | 27 |
| Common Lambda + K+ and integration tests | 480 | 25 |

All K+ kernel, zero-mode, determinism, symmetry-warning, axis-order,
non-even-fermion reflection, local/lattice transform, uniform fit, and config
tests pass. Small-Lambda single-pole time/frequency round trips meet `1e-12`.
The bosonic noisy-tau regression tests pass with K+ at both noise levels.

The final 25 failures comprise all 22 baseline failures and these three
additional failures already observed in the Lambda-only stage:

| Test | Observed | Required |
|---|---:|---:|
| `test_static_only_boson_is_split_as_c0` (legacy boson QP) | max difference `5.15e-6` | `atol=1e-6`, `rtol=1e-7` |
| `test_flocdyn_elastic_projection_accepts_noisy_uniform_data` | max independently recovered coefficient `7.58e-5` | `<=1e-5` |
| `test_fermion_noise_tau_regression` | max anti-Hermitian component `3.95e-6` | `<1e-6` |

These remain unresolved numerical regressions of the common bandwidth change.
No tolerances were relaxed. The full suite is **not green**.

## Validation boundary

The test16 replay and fresh-Prefix Grammar SCF require separate approval and
have not been run. No scheduler jobs, remote edits, plots, or Drive uploads were made.
Synthetic transform tests do not establish scientific or SCF convergence.

Start with a new Prefix. Existing HDF5 nodes/mixing history are not migrated;
historical-file postprocessing with the changed basis is not validated.
The odd K- sector, general complex/non-density channels, fermion symmetry,
static Sigma Hermiticity, and the polarization-assembly floor remain outside
this implementation.
