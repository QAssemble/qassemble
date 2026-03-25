# Hartree-Fock Approximation

## Overview

The Hartree-Fock (HF) method is a mean-field approximation that treats electron-electron interactions through static effective potentials. Rather than solving the full many-body problem, HF replaces the two-body interaction with an effective single-particle potential constructed self-consistently from the electron density. The total HF self-energy decomposes into two contributions: the Hartree (direct) term and the Fock (exchange) term.

## Hartree Self-Energy

The Hartree term represents the classical electrostatic potential felt by an electron due to the charge density of other electrons. It is given by:

$$
\Sigma^{H\, pr}_{ij\sigma}(\mathbf{k}) = \sum_{\sigma'} \sum_{q} \sum_{kl} V^{p\sigma;q\sigma'}_{ijkl}(\mathbf{k}=0) \times \frac{1}{N_\mathbf{k}} \sum_{\mathbf{k}'} n^{q}_{lk\sigma'}(\mathbf{k}') \delta_{pr}
$$

where $N_\mathbf{k}$ is the number of k-points in the Brillouin zone, $n^{q}_{lk\sigma'}(\mathbf{k}')$ represents the momentum-resolved density, and the interaction $V^{p\sigma;q\sigma'}_{ijkl}(\mathbf{k}=0)$ is evaluated at zero momentum transfer. This term is diagonal in the basis-site index ($\delta_{pr}$) and captures the average electrostatic repulsion.

In QAssemble, the `SigmaHartree` class (child of `FLatStc`) computes this quantity. It accepts the density matrix $n$ and the bare interaction $V$ as inputs and returns $\Sigma^H$ as an `FLatStc` object.

## Fock Self-Energy

The Fock term accounts for exchange interactions arising from the antisymmetry of the fermionic wavefunction, providing a non-local correction:

$$
\Sigma^{F\, p,q}_{ij\sigma}(\mathbf{k}) = -\sum_{\mathbf{R}} \sum_{kl} n^{q,p}_{lk\sigma}(\mathbf{R}) \times V^{p\sigma;q\sigma'}_{ijkl}(\mathbf{R}) \delta_{\sigma\sigma'} e^{i\mathbf{k}\cdot\mathbf{R}}
$$

where $n^{q,p}_{lk\sigma}(\mathbf{R})$ is the density matrix in real space connecting sites separated by lattice vector $\mathbf{R}$, and $\delta_{\sigma\sigma'}$ ensures that exchange occurs only between electrons of the same spin. This term is generally non-diagonal in both real space and orbital indices, and is responsible for phenomena like exchange splitting and magnetic ordering.

In QAssemble, the `SigmaFock` class (child of `FLatStc`) accepts $n$ and $V$ as inputs and returns $\Sigma^F$ as an `FLatStc` object.

## HF Hamiltonian and Self-Consistent Field Procedure

The total HF self-energy combines both contributions:

$$
\Sigma^{HF} = \Sigma^H + \Sigma^F
$$

The effective single-particle HF Hamiltonian is then:

$$
H_{HF}(\mathbf{k}) = H_0(\mathbf{k}) + \Sigma^{HF}(\mathbf{k}) - \mu \hat{I}
$$

where $\mu$ is the chemical potential of the system and $H_0$ is the non-interacting Hamiltonian constructed from hopping amplitudes and on-site energies.

Since both $\Sigma^H$ and $\Sigma^F$ depend on the density matrix $n$, which is itself determined by $H_{HF}$, the problem must be solved self-consistently. The SCF procedure proceeds as follows:

1. **Initialize**: Construct the non-interacting Hamiltonian $H_0$ from hopping amplitudes $t$ and on-site energies $\epsilon$ (`NIHamiltonian` class). Construct the bare Coulomb interaction $V$ (`VBare` class).

2. **Compute self-energies**: Given the current density matrix $n$, evaluate $\Sigma^H$ (`SigmaHartree`) and $\Sigma^F$ (`SigmaFock`).

3. **Update Hamiltonian**: Form $H = H_0 + \Sigma^H + \Sigma^F$ in the `Hamiltonian` class.

4. **Adjust chemical potential**: Search for $\mu$ such that the target electron filling $N_e$ is satisfied.

5. **Compute density**: Evaluate the Fermi-Dirac distribution $n = 1/(e^{\beta(H-\mu)} + 1)$ to obtain the updated density matrix.

6. **Check convergence**: Compare the new density matrix with the previous iteration. If converged, stop; otherwise, apply density mixing (`OccMixing`) and return to step 2.

The `Hamiltonian` class acts as the central hub for this workflow, aggregating the self-energy contributions and managing the chemical potential search via `CalMu0`, `SearchMu`, and `UpdateMu`.

## Bare Coulomb Interaction

The `VBare` class constructs the bare Coulomb interaction $V$ from user-specified parameters. Interactions are handled in both local and non-local forms:

- **Local interactions**: Specified through Slater or Kanamori parameterizations, with support for transformations between the two. The `VLoc` class in `BLocStc` provides `SlaterKanamori`, `SlaterParameter`, and `KanamoriParameter` methods.
- **Non-local interactions**: Specified either explicitly through site-to-site couplings $V_{ij}$ or generated from model potentials:
    - **Ohno**: $V(r) = U / \sqrt{1 + (Ur/e^2)^2}$, interpolating between on-site $U$ and long-range $e^2/r$ Coulomb behavior.
    - **JTH**: Adopts the same functional form as Ohno but allows a consistent on-site Coulomb value across equivalent orbitals.

## References

- D. R. Hartree & W. Hartree, *Proc. R. Soc. A* **150**, 9-33 (1935).
- J. C. Slater, *Phys. Rev.* **32**, 339 (1928); **35**, 210 (1930); **81**, 385 (1951).
- C. Froese Fischer, *Comput. Phys. Commun.* **43**, 355 (1987).
- J. C. Slater, *Quantum Theory of Atomic Structure* (McGraw-Hill, 1960).
- J. Kanamori, *Prog. Theor. Phys.* **30**, 275 (1963).
- D. Van Der Marel & G. A. Sawatzky, *Phys. Rev. B* **37**, 10674 (1988).
- H. U. R. Strand, *Phys. Rev. B* **90**, 155108 (2014).
- K. Ohno, *Theor. Chim. Acta* **2**, 219 (1964).
