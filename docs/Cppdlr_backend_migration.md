# Native cppdlr K+ boson backend

Updated 2026-10-27 from `f20a102` on `DLR_GW_EDMFT_nocausal`.
`BosonSymmetry=True` now uses `pycppdlr.PairedBoson`. Fermions default to
`Basis(..., statistic="fermion", symmetrize=True)` (`FermionSymmetry=True`);
`FermionSymmetry=False` and `BosonSymmetry=False` use unsymmetrized Basis.
The local binding
must be rebuilt; see [installation](../pycppdlr/README.md).

## Kernel and node definition

For a positive dimensionless pole `x`, the native time kernel is
`k_it(t,x) + k_it(t,-x)`, and the frequency kernel is
`Re[k_if(n,x) + k_if(-n,x)]`. Thus each column is already reflection-even.
This uses the sum normalization; an averaged image kernel spans the same
sector with rescaled coefficients. There are no redundant +/-pole coefficient
pairs and no post-solve coefficient averaging.

The pole candidates are zero plus the positive half of cppdlr's fine grid.
`pivrgs(weighted_kernel, eps)` selects poles using cppdlr's existing
sqrt-quadrature weights on the full relative-time fine grid. Independent
time rows come from the first fine-grid half, and independent frequency rows
from integers `0..nmax`; both use `pivrgs(..., 1e-100)`. Both transforms solve
square systems with native `nda::lapack::getrf/getrs`, without SVD.

**Zero is a candidate, not a forced selected pole.** This follows the explicit
decision to preserve the scratch algorithm. In the measured `Lambda=50000`
basis the first pole is `0.0016335119334385077`, not zero. If zero is selected,
its kernel is the single native kernel: `-1/2` in time and
`-beta/2 * delta(n,0)` in frequency. The adapter reverses the sign, so a
selected zero coefficient of 2 represents constant time data of 1.
The raw zero-kernel convention is tested independently of pole selection.

With coefficient rank `r`, exposed time samples contain `2r` or `2r-1` rows;
exposed frequency samples contain `2r-1` rows, including zero. Time and
frequency samples include both mirror partners. The low-level `PairedBoson`
solves only independent half rows and preserves complex data by solving real
and imaginary RHS separately. It does not project samples.

The QAssemble adapter projects samples onto the element-wise real/even sector
before solving. Odd/imaginary residuals above `max(100*eps, 1e-10)` warn.
The existing uniform-grid SVD data fit is retained with all signed rows;
it is separate from the native node transforms.

## Interface and dtype contracts

Public adapter nodes remain ascending absolute times and integer Matsubara
indices. Relative-time coordinates are retained inside the native kernels.
Coefficients obey `c_adapter = -c_native`, and beta is applied once.
Tau transforms and frequency inverses preserve trailing axes; rank-3 frequency
forward/evaluation retains the legacy transpose. Frequency evaluation retains
the legacy whole-output conjugation. Symmetric K+ evaluation accepts only
Matsubara points in the native integer range; other complex points raise
`ValueError`. Nonsymmetric real-axis evaluation is unchanged.

`DLR.BF2T` and `BT2F` now always return `complex128`, including real/even
bosonic data. Their batched and local/lattice callers inherit this contract.
This fixes `BLatDyn.K2R` failing when it multiplies the returned array by
complex phases in place. `K2R` itself is unchanged. Symmetric fermions retain
complex coefficients and off-axis evaluation. Adapter `symmetrize` selects
mirror nodes, whereas `paired` identifies K+ bosons only; symmetric bosonic
causal QP remains unsupported on this branch.

## Symmetric fermion default and restart compatibility

`FermionSymmetry=True` selects mirror-closed poles, tau nodes, and fermionic
Matsubara indices (`n <-> -n-1`). This is grid symmetry only, without coefficient
projection or tau Hermitian averaging. Stable Hermitian-fold sorting preserves
original positive samples when a mirrored sample duplicates their frequency.

The user-supplied test18 iteration-1 production replay changed only fermion
symmetry: max Im G decreased from 1.17e-4 to 1.59e-6, and Im n from 4.7e-6
to 1.7e-8. Sigma_imp differed by 0.13 CTQMC sigma and mu1 by 2.4e-5
(symmetric mu1: 2.6399771325). These are supplied prior replay measurements,
not a rerun of the implementation here or multi-iteration convergence evidence.
The approximately 1e-5 real-part error is outside this change's scope.

Changing either symmetry flag changes the DLR nodes; old HDF5 node data and
mixing histories cannot be restarted, so start with a new Prefix.

## Fresh local synthetic comparison

Python 3.13.9, one OMP/OpenBLAS thread, `eps=1e-14`, `beta=100`,
`Lambda=50000`. The comparison freezes the old adapter from `git show f20a102`
and does not replace that arm with the new implementation. Dense time points
combine 401 uniform points with 60 logarithmic points near each endpoint.
Frequency indices combine -4000..4000 with logarithmic tails up to 3e6.
Errors are `max(abs(actual-exact))/max(abs(exact))` per output domain.

The analytic paired-pole functions are

```text
T(u) = sum_j a_j [exp(-u*x_j)+exp(-(1-u)*x_j)]/[1+exp(-x_j)]
F(n) = sum_j a_j beta*2*x_j*tanh(x_j/2)/[(2*pi*n)^2+x_j^2].
```

| i-nu -> tau relative error | f20a102 | RS (SVD) | RS-LU | Native K+ |
|---|---:|---:|---:|---:|
| In-band: x=[50,300,2000,49000], a=[.5,1,2,5] | 2.426e-11 | 5.050e-12 | 2.995e-11 | **3.2566e-12** |
| x=[18550,.913], a=[.7,.3] | 1.866e-11 | 2.717e-11 | 5.706e-12 | **1.7193e-12** |
| Near zero: x=[.01,300,2000], a=[1,1,1] | 5.691e-11 | 6.859e-12 | 6.354e-11 | **1.5558e-13** |

For the in-band function, native K+ errors in the four directions
tau->tau, tau->frequency, frequency->tau, frequency->frequency are
`[3.5563e-12, 7.3987e-14, 3.2566e-12, 8.2197e-15]`.
Its coefficient maxima are `5.35` from tau and `5.37` from frequency.
The repository and scratch native-LU arms select identical nodes and have
zero difference in every reported metric across all four clean cases,
including a deliberately out-of-band control.

RS retains redundant kernel images and uses SVD; RS-LU retains the original
cppdlr nodes and restricts to square blocks. Native K+ selects nodes on the
paired kernels themselves. RS-LU is about 9.2 times worse on the in-band case
and 408 times worse on the near-zero case; the second table row has a smaller
factor. These comparisons support the selected clean-function path, not a
universal superiority claim. SciPy/native LU have comparable function errors;
their ill-conditioned coefficients need not agree bit for bit.

The Lambda=10000 regression uses x=[10,60,400,9800], a=[.5,1,2,5].
Native K+ frequency->tau error is `6.63e-13`, maximum coefficient `5.33`;
f20a102 gives `3.60e-11` and `13.69`. The test requires all four errors below
`1e-11` and both coefficient maxima below 10. This clean-data coefficient bound
does not bound coefficients of noisy CTQMC data.

The `1e-12` target is still exceeded by the Lambda=50000 in-band function.
The out-of-band control at E=1500 (cutoff=500) has native K+ frequency->tau
error `2.26e-3` and coefficients around `8.24e8`; finite-band accuracy does
not extend to that input.

Current in-band diagnostic, from the repository root:

```bash
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
import numpy as np
from QAssemble.utility.DLR import CppdlrDLR
d = CppdlrDLR(50000, 1e-14, 1, True)
beta = 100.
x, a = np.array([50.,300.,2000.,49000.]), np.array([.5,1.,2.,5.])
def tau(u):
    return ((np.exp(-u[:,None]*x)+np.exp(-(1-u[:,None])*x))/(1+np.exp(-x))) @ a
def freq(n):
    return (beta*2*x*np.tanh(x/2)/((2*np.pi*n[:,None])**2+x*x)) @ a
u = np.unique(np.r_[np.linspace(0,1,401),np.geomspace(1e-8,.5,60),1-np.geomspace(1e-8,.5,60)])
c = d.dlr_from_matsubara(freq(d.dlrmf), beta)
print(np.max(abs(d.eval_dlr_tau(c,beta*u,beta)-tau(u)))/np.max(abs(tau(u))))
PY
```

## Prior test16 scratch replay: supplied evidence, not rerun here

The following rounded figures were supplied with the implementation plan.
The local checkout did not contain the corresponding remote replay JSONs.
They are reference targets for the separately approved repository-code replay,
not measurements of a new production run after this change.

| Metric | f20a102 scratch arm | K+ scratch arm |
|---|---:|---:|
| max abs Sigma_GW coefficient | 2.6e7 | 2.8e5 |
| max abs W_c coefficient | 4.5e8 | 1.4e7 |
| Off-node tau Sigma_GW relative error | 3.4e-5 | 1.2e-6 |
| P3 node-pair metric (`p3.pair.abs`) | not supplied | 3.6e-10 |

Coefficient size depends on basis normalization; the off-node function metric
is necessary alongside coefficient comparisons. The static Sigma_H analysis
and G2 floor are outside this implementation.

## Verification and deployment boundary

The freshly measured utility baseline is **515 passed / 23 failed**, rather
than the 9 failures anticipated by the plan. Dtype and binding commits retain
those exact 23 failing IDs. After the K+ switch, the suite is **517 passed /
23 failed**, with two new tests and zero newly failing IDs. Failing-ID equality
does not establish identical residuals in projection-containing tests.
The adapter/symmetric subset has 63 passes.

The full native binding suite is **125 passed / 4 failed**. All four failures
are the existing direct `Basis(symmetrize=True)` large-cutoff accuracy tests;
they remain because that API is preserved. The fast native suite, including
PairedBoson, has 104 passes. OMP/OpenBLAS 1/8 determinism tests include K+.
No existing tolerance was relaxed.

Fermion and nonsymmetric boson nodes and all six transform/evaluation outputs
match frozen f20a102 exactly in the compatibility probe. An additional cache
fixture was changed to use actual boson Matsubara points rather than fermion
points when selecting the K+ evaluator. The uniform-fit algorithm is unchanged.

Rebuild pycppdlr and use a **fresh Prefix**: this replacement changes nodes
again, so f20a102 HDF5 data and mixing history cannot be restarted. There is
no automatic migration or backend fallback. `CLAUDE.md` is updated locally and
retains its pre-existing ignored status. The unrelated scratch-analysis doc
and dirty cppdlr submodule are preserved.

Grammar build/replay and multi-iteration SCF require the separately requested
approval. They have not been executed for these new K+ commits. Future submitted
jobs must retain completion/failure email notifications. Scientific convergence
and the proposed iteration plots remain unverified.
