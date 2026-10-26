# cppdlr backend migration

All three QAssemble paths now use `pycppdlr.Basis`: fermion with
`symmetrize=False`, and boson with `symmetrize=BosonSymmetry`. The local binding
must be installed first; see [installation](../pycppdlr/README.md).
Neither the runtime nor the utility tests require pydlr.

## Adapter contracts

`CppdlrDLR` exposes ascending absolute-time nodes, ascending Matsubara integer
indices, and ascending dimensionless poles. Node input is reordered for native
operators; output is reordered back. Coefficients have the pydlr sign:
`c_adapter = -c_cppdlr`. Native transforms apply beta and the bosonic tanh
correction once. Effective `eps = max(requested_eps, 1e-14)`.

Tau transforms and Matsubara inverses preserve trailing axes. Matsubara forward
and arbitrary-frequency evaluation swap the two trailing axes only for rank-3
arrays, preserving the historical interface. Arbitrary-frequency evaluation
retains pydlr's anti-linear convention:

```text
eval(c,z) = conjugate(sum_x c_x * correction_x / (z + pole_x/beta))
```

For Matsubara points this calls native evaluation at `-n-1` (fermion) or `-n`
(boson), then applies the historical transpose/conjugation. Other complex
frequencies use the same explicit rational kernel. Node reconstruction uses
native `coef_to_tau`, preserving its relative-coordinate precision. Absolute
tau evaluation maps `u=tau/beta` to `u-1` for `u>1/2`, preserving `u=1` as the
right endpoint. Absolute node sampling can still lose endpoint precision at
large bandwidths, as explained in the binding README.

Symmetric boson nodes alone do not enforce the element-wise real/even sector.
The adapter averages real mirror samples before solving, warns on discarded
odd/imaginary residuals, and averages recovered real coefficients over exact
opposite-pole pairs. Coefficient evaluation also applies this projection, so
uniform-grid fitting retains the real/even sector. This additional coefficient
projection corrects parity breaking by native LU/GELSS. Symmetric bosons have
`rank` tau nodes, `rank+1` Matsubara nodes (including zero), and no zero pole.
The single-pole causal QP remains explicitly unsupported for this basis.

## Step 0 measurements, 2026-10-25

Python 3.13.9, installed pycppdlr from the native binding implementation at
QAssemble commit `1b4347a`; cppdlr 1.3.0. Measurements used one OpenBLAS thread
unless indicated. The pre-existing dirty cppdlr ChangeLog was preserved.

Unit-coefficient time/frequency kernels confirmed the sign, beta factor,
bosonic tanh correction and integer-index conventions for both statistics with
symmetrization on and off. Maximum frequency-kernel absolute difference at
`Lambda=50000` was `4.58e-16`; the time-kernel comparison was exact.
Symmetric time nodes were exact mirror pairs at the tested bandwidths.

OMP 1 versus 8 gave identical bytes for poles, native relative-time nodes,
integer frequency nodes, and unit-coefficient tau/frequency arrays for all four
statistic/symmetrization combinations at `Lambda=100` and `50000`. At `100`, a
comparison changing both OMP and OpenBLAS threads from 1 to 8 also passed.
This is same-build determinism, not a cross-platform guarantee.

The large-bandwidth comparison used `Lambda=50000`, `eps=1e-14`, `beta=7.3`,
two even pole pairs at dimensionless energies `+/-18550` and `+/-0.913`, with
pair weights `0.7` and `0.3`. Errors below are maximum absolute error divided by
the maximum absolute analytic value in the output domain. Evaluation used 201
uniform absolute-time points and Matsubara indices -157 through 157. The old K+
implementation was measured before removal, with the same eps and function.

| Input/output | Previous K+ | Native cppdlr | cppdlr plus coefficient parity projection |
|---|---:|---:|---:|
| tau -> tau | 7.88e-15 | 1.41e-13 | 1.41e-13 |
| tau -> frequency | 2.01e-15 | 1.87e-15 | 1.87e-15 |
| frequency -> tau | 1.33e-13 | 2.10e-11 | 1.79e-11 |
| frequency -> frequency | 3.38e-15 | 9.91e-15 | 8.66e-15 |

**The frequency-to-tau error exceeds the unchanged 1e-12 target.** Pair
projection restores real/even symmetry but does not cure the native accuracy
limit. The user reviewed these measurements and authorized completing the
migration with this limitation documented. No tolerance was relaxed.

The current adapter diagnostic is reproducible from the repository root:
The simplified analytic formula below gives about `1.81e-11` on this build;
roundoff differences in the sampled inputs affect this ill-conditioned solve.

```bash
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
import numpy as np
from QAssemble.utility.DLR import CppdlrDLR
beta = 7.3
d = CppdlrDLR(50000, 1e-14, xi=1, symmetrize=True)
x, weights = np.array([18550., .913]), np.array([.7, .3])
def tau(u):
    return ((np.exp(-u[:, None]*x) + np.exp(-(1-u[:, None])*x))
            / (1+np.exp(-x))) @ weights
def frequency(n):
    return (beta * 2*x*np.tanh(x/2)
            / ((2*np.pi*n[:, None])**2+x*x)) @ weights
c = d.dlr_from_matsubara(frequency(d.dlrmf), beta)
u = np.linspace(0, 1, 201)
error = np.max(abs(d.eval_dlr_tau(c, u*beta, beta)-tau(u))) / np.max(abs(tau(u)))
print('frequency-to-tau relative error:', error, 'target:', 1e-12)
PY
```

## Validation and boundaries

The requested Python 3.13.9 utility suite, with `PYTHONPATH=$PWD/src` and one
OMP/OpenBLAS thread, changed from **480 passed / 25 failed** before the migration
to **515 passed / 23 failed** afterward. The new backend test file has 33 passing
cases. Failure-ID comparison found zero new failures and these two resolved IDs:

- `test_fermion_causal_projection.py::test_flocdyn_elastic_projection_accepts_noisy_uniform_data`
- `test_uniform_to_dlr_interp.py::test_fermion_noise_tau_regression`

The remaining 23 baseline failures are in boson causal projection (18), boson
static fitting (1), GW+EDMFT composition (2), and local-method result types (2).
No existing tolerance was relaxed. The full suite remains red because of these
baseline failures. Shared failing IDs do not establish unchanged numerical
residuals: QP-containing tests exercise causal projection as well as the basis,
and their input/output differences are not standalone DLR transform errors.
QP solver optimality was not established by this migration validation.
Validation also imported DLR/Causal and exercised all three
runtime paths with imports of `pydlr` deliberately blocked. Changed Python and
project metadata parse, and `git diff --check` passes.

`README.md` and the local `CLAUDE.md` contain installation/restart guidance;
`CLAUDE.md` remains untracked by the repository's pre-existing ignore rule.
Historical diagnostic notebooks were not executed or migrated.

`test_cppdlr_backend.py` checks analytic off-basis multipole Hermitian matrices
on common tau and Matsubara points for all three runtime paths. Unit-pole
coefficient functions check arbitrary complex frequencies, including real-axis
`z+i*eta`, the legacy transpose/conjugation distinction, native delegation,
coefficient sign, beta scaling, endpoint handling, and invalid inputs.

An imaginary-axis DLR fit does not guarantee accurate analytic continuation
of off-basis poles near the real axis. The complex-frequency tests establish
the rational evaluator's convention using known coefficients; they do not
claim physical real-axis accuracy from an imaginary-axis fit. The existing
`FLatDyn.AnalyticContinuation` docstring/code sign discrepancy is outside this
migration; its runtime convention was preserved.

Start with a **new Prefix**. Existing pydlr/K+ HDF5 samples, checkpoints and
mixing histories use incompatible nodes and cannot be reused. There is no
automatic restart migration or backend fallback. Physical GW/EDMFT comparison,
Grammar compilation/deployment and job submission remain outside this validation.
