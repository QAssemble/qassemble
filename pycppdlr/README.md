# pycppdlr

Minimal, standalone NumPy bindings for cppdlr's pole/node selection, transforms,
and evaluation. QAssemble is not imported or modified. This is not a pydlr adapter.

Only builds from the QAssemble checkout with its sibling `cppdlr/` submodule
are supported. SDists and portable binary wheels are not supported yet.
The local wheel requires the linked Homebrew libraries on the same machine.

## Build on macOS arm64

Required: CMake >=3.22, a C++23 compiler/standard library, OpenMP, OpenBLAS,
MPI and HDF5 C/HL. The validated compiler is Homebrew LLVM; AppleClang is not
a supported target of this initial build. No compiler or system library is installed
by this package. NumPy is the only Python runtime dependency.

```sh
git submodule update --init cppdlr
CXX=/opt/homebrew/opt/llvm/bin/clang++ \
  ~/.pyenv/versions/3.13.9/bin/python -m pip install -v ./pycppdlr \
  -Ccmake.define.CMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
python -m pytest pycppdlr/tests -m 'not slow' -q
python -m pytest pycppdlr/tests -m slow -s -q
```

The build backend and nanobind are installed in pip's isolated build environment.
On this host LLVM 21.1.2 with the default macOS 27 SDK fails even a standalone
`#include <complex>` (`NAN`/`INFINITY` missing); the installed 26.5 SDK passes.
The command selects that SDK without changing compiler/system headers.
GTest and fmt are skipped only because this extension disables upstream tests and
does not install upstream CMake exports. No cppdlr source patch is applied.

| Source | Version | Pinned peeled commit |
|---|---|---|
| cppdlr | 1.3.0 | `5f9adf91d7620ce145ead8e4e906151493038af3` |
| itertools | 1.3.0 | `2bb4da2b56ab49b3cd7b5f27cfb88c5320fe8524` |
| mpi | 1.3.0 | `c745cd5049510c0fb981042dde6b34b6d8c346ab` |
| h5 | unstable, 2026-06-24 | `fca9584b2390745cad33fc513094d35921ccdbe2` |
| nda | unstable, 2026-06-24 | `826bca37d014b82beaf15d6b89d3ec20cd9b82cb` |

nda 1.3.0 (`39f6e149507860cfb286b24a58fcbb0932fa2586`) was tried and
failed: cppdlr requires `nda::linalg::matmul/matvecmul`, absent in that tag.
The fallback above is the last first-parent unstable commit before 2026-06-26 UTC.
That nda also requires h5's newer shape-pointer array API; h5 1.3.0 failed to
compile, so its matching June unstable revision is pinned too.
HDF5, MPI and OpenMP support remain enabled explicitly.

After an initial build, repeat the pip command with
`-Ccmake.define.FETCHCONTENT_FULLY_DISCONNECTED=ON` to rebuild using the populated
dependency sources in `pycppdlr/build/`. This disables dependency fetching;
pip may still need its own cached build requirements.

## Native API and convention

All transforms and kernel evaluations call cppdlr `imtime_ops`/`imfreq_ops`
methods, as TRIQS does. Python contains no kernel implementation, linear solve,
least-squares solver, or matrix transform. It validates/reshapes arrays and restores
their trailing axes. The diagnostic test alone retains a NumPy solve for comparison.

```python
import pycppdlr
b = pycppdlr.Basis(100., 1e-14, statistic="boson", symmetrize=True)
# Sample data in this order, using native relative times b.t_rel.
c = b.tau_to_coef(values_on_b_t_rel)
freq = b.coef_to_freq(c, beta=20.)
values = b.eval_tau(c, [0., -0.1, 1.])
```

`Basis.it` and `Basis.ifreq` retain the native operator objects, including prepared
LU factors. The first data axis is nodes/coefficients; arbitrary trailing axes are
flattened to a batch, converted to contiguous complex128, then restored. There is
no transpose/conjugation. Real input returns real output only when the native
output's imaginary part is exactly zero.

| Public API | Native operation |
|---|---|
| `build_dlr_rf(lamb, eps, symmetrize=False)` | `build_dlr_rf` |
| `ImTime(lamb, rf, symmetrize=False)` | Persistent `imtime_ops` |
| `ImTime.rank`, `ImTime.nodes` | `rank()`, copied `get_itnodes()` |
| `ImTime.vals2coefs(g)`, `coefs2vals(c)` | Same native methods |
| `ImTime.coefs2eval(c, t_rel)` | Native pointwise evaluation in a C++ loop |
| `ImFreq(lamb, rf, boson, symmetrize=False)` | Persistent `imfreq_ops` |
| `ImFreq.rank`, `ImFreq.nodes` | `rank()`, copied integer `get_ifnodes()` |
| `ImFreq.vals2coefs(beta, g)`, `coefs2vals(beta, c)` | Same native methods; beta handled inside cppdlr |
| `ImFreq.coefs2eval(beta, c, n)` | Native pointwise evaluation in a C++ loop |

The raw classes accept 2D contiguous complex128 data, float64 poles/relative times,
and contiguous int64 evaluation indices. Raw frequency indices are not implicitly
cast: fractional arrays raise TypeError. Raw poles must be finite, strictly
increasing, within the cutoff, and exactly paired with even length when symmetrized.
All returned arrays own copied buffers independent of the native objects. NDA
bounds checks remain enabled to turn rank-degenerate raw inputs into exceptions.

`Basis` exposes `rf`, `rank`, `t_rel`, `n`, and a display-only absolute time
array `u`. Its methods are `tau_to_coef(values)`, `coef_to_tau(c)`,
`freq_to_coef(values, beta=1)`, `coef_to_freq(c, beta=1)`,
`eval_tau(c, t_rel)` and `eval_freq(c, n, beta=1)`.

**Time coordinates changed from the initial wrapper:** node order is cppdlr's
native order, never sorted. Sample and evaluate using `t_rel`, in [-0.5, 0.5],
with +1 reserved for the right endpoint. Negative t means distance from the right
endpoint. `u=where(t_rel<0, 1+t_rel, t_rel)` is for display only: forming 1+t
loses endpoint precision and must not be used for sampling/evaluation. Endpoint
+1 is preserved; it is not mapped to zero. Frequency indices must be integers in
[-1073741824, 1073741823], so upstream's 32-bit `2*n+1` arithmetic cannot overflow.

For u=tau/beta and w=beta*E, cppdlr's native kernels are:

```text
A_tau(u,w) = -exp(-u*w)/(1+exp(-w))           (native relative evaluation is stable)
A_F(n,w)   = 1/(i*(2*n+1)*pi-w)
A_B(n,w)   = tanh(w/2)/(i*2*n*pi-w)          (n=w=0: -1/2)
```

The time kernel has the opposite sign to pydlr. Bosonic tanh is already included.
Physical frequency values contain beta exactly once, applied by cppdlr.
Square transforms use the native prepared LU. Symmetric boson frequency fitting
uses cppdlr's own GELSS with rcond=0; the previous NumPy cutoff is removed.

Symmetric nodes/poles do not project data into the real/reflection-even K+ sector.
TRIQS explicitly labels symmetrize **Experimental!**
([pinned source](https://github.com/TRIQS/triqs/blob/b85f3d131d6a20fe59ce224ffe4c1319b6dc4676/c%2B%2B/triqs/mesh/dlr_imtime.hpp#L60)).
Use **eps >= 1e-14 for symmetric modes** as a minimum precaution, not an accuracy
guarantee. eps=1e-15 remains a stress case. K+ projection, exact constant-time
zero mode handling, pydlr adaptation, MPI rank-0 generation, and the QAssemble eps
default remain outside this change.

## Native validation (2026-10-23)

Host: Apple M3 arm64, Python 3.13.9, NumPy 2.3.4, Homebrew LLVM 21.1.2,
macOS SDK 26.5, OpenBLAS 0.3.28, Open MPI 5.0.7, HDF5-MPI 1.14.6.

- Fast suite: **87 passed**, covering independent analytic kernels/functions,
  complex/noncontiguous batches, copied-buffer lifetime, invalid inputs, native
  integer limits, right endpoint, display-only u, real-output policy, and
  successful transforms with NumPy solve/lstsq monkeypatched to fail.
- Same-build determinism passed for all OMP/OpenBLAS thread combinations in
  {1,8}, comparing exact bytes for poles, native nodes and unit-coefficient
  native transforms.
- Slow suite: **21 passed, 4 failed** (24 benchmark cases plus one print-only
  diagnostic). Unchanged tau-origin <=1e-12 assertions fail for symmetric
  eps=1e-14 at Lambda=10000 and 50000, both statistics. No threshold was
  relaxed and no failing condition was skipped. These are now native cppdlr
  transforms evaluated on native relative coordinates.
- All 12 eps=1e-15 stress constructions completed; no accuracy bound is imposed.
  The largest tau-origin error is about 2.88e-10 at Lambda=50000/symmetric boson.
  The initial Python path had reached 1.08e-7; both the solver and the treatment
  of time coordinates have changed, so that improvement is not attributable
  solely to replacing NumPy solve.
- QAssemble utility suite: **480 passed, 25 failed**, with the same failing test
  IDs as the archived clean HEAD baseline. QAssemble source was not modified.
- Offline dependency-source wheel rebuild passed. The wheel contains only
  __init__.py and _core*.so plus normal dist-info, with no upstream headers or
  static libraries. It installed in a new venv with NumPy 2.5.4, no SciPy or
  QAssemble, and passed native analytic fermion/boson transform checks.

### Solver and sampling diagnostic

At Lambda=50000, eps=1e-15, symmetric boson, a complex multi-pole function is
sampled on the same relative nodes. Native and NumPy solves receive the identical
matrix/data; all reconstructions below use the new native evaluator.

| Path | Relative tau error | Relative frequency error |
|---|---:|---:|
| Native solve, stable relative input | 1.541e-13 | 3.806e-14 |
| NumPy solve, same stable input | 4.550e-13 | 1.185e-13 |
| NumPy solve, legacy absolute-time input | 5.696e-11 | 1.495e-11 |

The relative/absolute sampling values differ by at most
5.818e-13. Native/NumPy coefficients differ by up to
75.790, yet their stable-input reconstructed functions are much closer.
The legacy row emulates the previous sampling/solve only, not its entire Python
evaluation path. This diagnostic isolates sensitivity without assigning every
previous error to NumPy or to cppdlr.

A standalone C++ executable (no Python or nanobind) independently reproduces
the symmetric boson tau-to-frequency target misses with native
`k_it -> vals2coefs -> coefs2eval`: 1.9793e-12 at Lambda=10000 and
2.3532e-11 at Lambda=50000 (eps=1e-14, beta=7.3, one off-basis pole).
This establishes a limitation of those upstream native-call conditions rather
than a remaining NumPy solve in the wrapper. Local reproduction artifacts are
`build/native_reference.cpp`, `build/native_reference-build.log`, and
`build/native_reference.log`.

### Construction and accuracy measurements

Single fresh subprocess per case, OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1, NDA
bounds checks enabled. Time covers Basis construction. macOS /usr/bin/time -l
peak RSS covers the worker, including imports and subsequent accuracy checks.
Errors are max absolute error divided by max absolute analytic value in each
output domain, over single/multiple off-basis poles, 200 relative-time evaluation
points, n=-157..157, beta=7.3. Each error column takes the worst output domain
and pole case. Tau-origin is the acceptance path; frequency-origin is diagnostic.
Runtime/memory acceptance limits remain undecided.

| Lambda | Statistic | Sym | eps | Rank | Build s | Peak RSS MiB | Tau-origin error | Freq-origin error |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| 1000 | fermion | off | 1e-14 | 67 | 0.120 | 50.1 | 3.77e-13 | 1.76e-12 |
| 1000 | fermion | off | 1e-15 | 70 | 0.130 | 49.1 | 2.82e-14 | 7.95e-14 |
| 1000 | fermion | on | 1e-14 | 66 | 0.100 | 48.7 | 2.62e-13 | 1.59e-13 |
| 1000 | fermion | on | 1e-15 | 72 | 0.110 | 49.3 | 2.34e-15 | 6.03e-14 |
| 1000 | boson | off | 1e-14 | 67 | 0.121 | 48.8 | 3.82e-13 | 4.40e-12 |
| 1000 | boson | off | 1e-15 | 70 | 0.130 | 49.1 | 3.02e-14 | 6.14e-14 |
| 1000 | boson | on | 1e-14 | 66 | 0.098 | 49.4 | 2.62e-13 | 3.05e-11 |
| 1000 | boson | on | 1e-15 | 72 | 0.111 | 49.5 | 1.87e-15 | 6.54e-12 |
| 10000 | fermion | off | 1e-14 | 95 | 1.213 | 106.3 | 8.32e-13 | 1.44e-11 |
| 10000 | fermion | off | 1e-15 | 100 | 1.381 | 109.6 | 8.46e-13 | 6.82e-13 |
| 10000 | fermion | on | 1e-14 | 96 | 1.187 | 107.0 | 1.90e-12 | 1.89e-11 |
| 10000 | fermion | on | 1e-15 | 102 | 1.329 | 111.0 | 4.63e-13 | 1.23e-12 |
| 10000 | boson | off | 1e-14 | 95 | 1.220 | 106.3 | 8.38e-13 | 6.40e-12 |
| 10000 | boson | off | 1e-15 | 100 | 1.373 | 109.7 | 7.70e-13 | 8.35e-13 |
| 10000 | boson | on | 1e-14 | 96 | 1.204 | 107.4 | 1.98e-12 | 5.69e-11 |
| 10000 | boson | on | 1e-15 | 102 | 1.346 | 111.5 | 4.40e-13 | 3.04e-11 |
| 50000 | fermion | off | 1e-14 | 114 | 7.965 | 400.2 | 1.29e-13 | 6.46e-12 |
| 50000 | fermion | off | 1e-15 | 122 | 9.008 | 424.5 | 1.40e-14 | 5.12e-13 |
| 50000 | fermion | on | 1e-14 | 116 | 7.808 | 406.2 | 2.72e-11 | 9.04e-10 |
| 50000 | fermion | on | 1e-15 | 120 | 8.602 | 418.5 | 2.86e-10 | 9.24e-10 |
| 50000 | boson | off | 1e-14 | 114 | 7.958 | 400.2 | 1.29e-13 | 6.40e-13 |
| 50000 | boson | off | 1e-15 | 122 | 9.051 | 424.6 | 1.40e-14 | 8.94e-14 |
| 50000 | boson | on | 1e-14 | 116 | 7.870 | 406.8 | 2.35e-11 | 2.27e-11 |
| 50000 | boson | on | 1e-15 | 120 | 8.692 | 418.8 | 2.88e-10 | 1.10e-11 |

Large-cutoff symmetric accuracy acceptance remains unresolved. Building and
installing a wheel does not establish production numerical validity.
