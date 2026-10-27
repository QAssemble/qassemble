# Symmetric boson DLR: native cppdlr K+

Updated 2026-10-27 on `DLR_GW_EDMFT_nocausal`, replacing the f20a102
post-projection basis. Both statistics retain `Lambda=beta*cutoff`.
The nine backend/K+ commits are also ported to `DLR_GW_EDMFT`, where the
bosonic causal QP now supports this basis.

`BosonSymmetry=True` selects `pycppdlr.PairedBoson`: each positive-pole
column is the sum of the native kernel images at opposite poles, so it is
reflection-even by construction. Zero is a candidate with the single native
kernel, but is not forced into the pivrgs result. This preserves the scratch
selection after the user clarified the original zero-pole assertion.
The adapter applies real/even projection to samples before solving and warns
on discarded residuals. It does not average solved coefficients.

Native `cppdlr::pivrgs` selects poles on the sqrt-quadrature-weighted paired
fine kernel and selects independent tau/Matsubara rows. Native LAPACK
`getrf/getrs` solves the square half-grid systems. No SVD is used in these
node transforms. The existing signed-uniform-grid SVD data fit is retained.

For coefficient rank r, exposed tau nodes have 2r or 2r-1 rows and frequency
nodes have 2r-1 rows including zero. The raw zero-column normalization, if
selected, is -1/2 in time and -beta/2 at frequency zero. QAssemble coefficients
reverse the native sign. General complex input is preserved by PairedBoson;
QAssemble's symmetric adapter implements the element-wise real/even sector.

`BF2T`/`BT2F` return complex128 so their batched local/lattice consumers can
apply complex Fourier phases. `K2R`, `T2mT`, and the W-on-fermion-tau consumers
are unchanged. K+ bosonic causal QP supports `reflection_symmetry=True` only;
frequency evaluation accepts only Matsubara points. Fermion and nonsymmetric
boson paths retain their previous implementation.

## Causal QP moment normalization

K+ has no odd m1 sector. Constructing its projector with
`reflection_symmetry=False` raises `ValueError` with
`K+ basis has no odd (m1) sector; use reflection_symmetry=True`.
For dimensionless pole x, the single even constraint uses
`m2[j] = multiplicity[j] * tanh(x[j]/2) * x[j]/beta`, where multiplicity
is 2 for positive K+ poles and 1 for a selected zero pole. A positive column
sums the +/-pole images; the zero pole contributes zero to this tail moment.
Asymmetric columns retain multiplicity 1. The existing sign constraints,
Matsubara data-fit objective, hard/relaxed policy and tolerances are unchanged.

At Lambda=160, beta=20 and nu=2*pi*1e6/beta, the measured ratio of
nu^2*K_even to the single-pole moment is 1.0000000 for asymmetric columns
and 2.0000000 for K+. The corrected moment rows agree with the kernel tail
within 6.48e-10 relative (test tolerance 1e-8); removing the multiplicity
makes the K+ regression fail. Both coefficient signs are tested for exact
causal-input certification and hard-moment noncausal projection.

## Measured clean-function comparison

Lambda=50000, beta=100, eps=1e-14. Errors are normalized maxima on the same
dense off-node time grid; see the full [migration report](Cppdlr_backend_migration.md)
for functions, evaluation grids, all four directions and provenance.

| i-nu -> tau relative error | f20a102 | RS-LU | Native K+ |
|---|---:|---:|---:|
| In-band four-pole-pair function | 2.426e-11 | 2.995e-11 | 3.2566e-12 |
| Near-zero pole function | 5.691e-11 | 6.354e-11 | 1.5558e-13 |

All selected nodes and reported clean-function metrics match the scratch
native K+ implementation exactly. The in-band result still exceeds 1e-12;
this is not a uniform accuracy guarantee. The Lambda=10000 regression meets
1e-11 in all four directions and has maximum coefficient below 10, while
the frozen old implementation fails both frequency-origin bounds.

The plan also supplied prior test16 scratch replay targets: Sigma_GW
coefficient maximum 2.6e7 -> 2.8e5, W_c 4.5e8 -> 1.4e7, off-node Sigma_GW
relative error 3.4e-5 -> 1.2e-6, and K+ P3 node-pair error 3.6e-10.
These remote replay figures were not rerun or independently verified during
the local implementation; they are explicitly marked as supplied evidence
in the migration report. The repository-code Grammar replay is a later gate.

## Tests and boundaries

The earlier no-causal backend validation was 517 passed / 23 failed.
The separately measured causal-branch results are:

| `DLR_GW_EDMFT` state | Passed | Failed |
|---|---:|---:|
| Baseline `4368204` | 497 | 2 |
| Nine backend/K+ commits | 562 | 10 |
| K+ causal QP support | 575 | 4 |

All ten port-stage failure IDs match the supplied simulation. The six
symmetric-projector guard failures disappear, including X-only fallback.
The remaining failures are the original two HFLoc/GWLoc save-contract tests
and two asymmetric static-QP numerical checks. See the
[migration report](Cppdlr_backend_migration.md#remaining-static-qp-diagnostics)
for the historical-state diagnosis and proposed follow-up; neither static-fit
nor tolerances were changed. Eight focused moment/K+ tests pass. Twelve
asymmetric pre/post comparisons have byte-identical moment rows, coefficients
and outputs, including forced QP solves.

The native binding suite has 125 passes and the same four prior failures in
the preserved direct Basis(symmetrize=True) large-cutoff stress tests. The
resource benchmark requires execution outside the macOS sandbox because
`/usr/bin/time -l` queries `kern.clockrate`. No tolerances were relaxed.

Rebuild the binding and start with a fresh Prefix. Nodes and coefficient
counts change, and old HDF5/mixing history is not migrated. Grammar build,
replay, fresh multi-iteration SCF and iteration plots require separate approval.
No new K+ remote job was submitted here. Static Sigma_H/G2-floor analysis,
general odd/complex channel physics and the static-QP follow-up remain outside
this change.
