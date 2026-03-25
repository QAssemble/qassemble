# GW Approximation

## Overview

The *GW* approximation goes beyond Hartree-Fock by including frequency-dependent (dynamical) screening of the Coulomb interaction. The name "GW" reflects that the self-energy is constructed as a convolution of the Green's function $G$ and the dynamically screened interaction $W$. While HF treats the Coulomb interaction as static, the *GW* approximation replaces $V$ with a frequency-dependent screened interaction $W$ that accounts for the polarization response of the electron gas.

The *GW* self-consistent workflow extends the Hartree-Fock calculation by incorporating three additional classes -- `PolLat`, `WLat`, and `SigmaGWC` -- that work together with the existing `GreenInt`, `SigmaHartree`, and `SigmaFock` classes.

## Non-Interacting Green's Function

The calculation begins with the non-interacting Green's function $G_0$, constructed from the non-interacting Hamiltonian $H_0$. In imaginary time, $G_0$ is defined as:

$$
G^{p,q}_{0\,ij\sigma}(\mathbf{R}-\mathbf{R}', \tau - \tau') = -\langle T_\tau c^p_{i\sigma}(\mathbf{R}, \tau)\, c^{\dagger\, q}_{j\sigma}(\mathbf{R}', \tau') \rangle_0
$$

where $T_\tau$ is the imaginary-time ordering operator and $\langle \cdots \rangle_0$ is the expectation value in the grand-canonical ensemble of $H_0$. The Matsubara-frequency representation is obtained via Fourier transform:

$$
G^{pq}_{0\,ij\sigma}(\mathbf{k}, i\omega_n) = \frac{1}{N_\mathbf{k}} \int_0^\beta d(\tau - \tau') \sum_{\mathbf{R},\mathbf{R}'} G^{pq}_{0\,ij\sigma}(\mathbf{R}-\mathbf{R}', \tau-\tau') e^{i(\mathbf{k}\cdot(\mathbf{R}-\mathbf{R}') - \omega_n \tau)}
$$

QAssemble uses the discrete Lehmann representation (DLR) for compact and accurate representation of both imaginary-time and Matsubara-frequency Green's functions. The `GreenBare` class constructs $G_0$ from $H_0$ produced by the `NIHamiltonian` class.

## Dyson Equation

The interacting Green's function $G$ is obtained through the Dyson equation:

$$
G^{pq}_{ij\sigma}(\mathbf{k}, i\omega_n) = G^{pq}_{0\,ij\sigma}(\mathbf{k}, i\omega_n) + \sum_{r,s} \sum_{k,l} G^{pr}_{0\,ik\sigma}(\mathbf{k}, i\omega_n)\, \Sigma^{rs}_{kl\sigma}(\mathbf{k}, i\omega_n)\, G^{sq}_{lj\sigma}(\mathbf{k}, i\omega_n)
$$

where $\Sigma$ is the irreducible electron self-energy. In the *GW* approximation, the total self-energy includes three contributions:

$$
\Sigma = \Sigma^H + \Sigma^F + \Sigma^{C,GW}
$$

The `GreenInt` class solves the Dyson equation at each iteration, receiving $G_0$ from `GreenBare` along with $\Sigma^H$ from `SigmaHartree`, $\Sigma^F$ from `SigmaFock`, and $\Sigma^{C,GW}$ from `SigmaGWC`.

## Irreducible Polarizability

The irreducible polarizability $P$ is computed from the current Green's function by evaluating the two-particle correlation function (particle-hole bubble). In the Matsubara frequency domain:

$$
P^{p\sigma;q\sigma'}_{ijkl}(\mathbf{k}, i\nu_n) = \int_0^\beta d\tau \sum_{\mathbf{R}} G^{qp}_{ki\sigma'}(-\mathbf{R}, -\tau) \times G^{pq}_{lj\sigma}(\mathbf{R}, \tau) \delta_{\sigma\sigma'} e^{i(\mathbf{k}\cdot\mathbf{R} - \nu_n \tau)}
$$

where $i\nu_n = 2n\pi/\beta$ are bosonic Matsubara frequencies, and the integration over imaginary time $\tau$ performs the convolution of two fermionic Green's functions. The Kronecker delta $\delta_{\sigma\sigma'}$ indicates that the electron and hole must have the same spin in the non-interacting bubble.

The `PolLat` class (child of `BLatDyn`) computes this quantity. It accepts the interacting Green's function $G$ and produces the bosonic response function that describes how the electron density responds to screened perturbations.

## Screened Coulomb Interaction

The screened interaction $W$ accounts for how the bare Coulomb potential $V$ is reduced (screened) by the polarization of the surrounding electron gas. It is obtained by the Dyson equation:

$$
W^{p\sigma;q\sigma'}_{ijkl}(\mathbf{k}, i\nu_n) = V^{p\sigma;q\sigma'}_{ijkl}(\mathbf{k}) + \sum_{rs} \sum_{i'j'k'l'} V^{p\sigma;r\sigma'}_{ii'j'l}(\mathbf{k})\, P^{r\sigma;s\sigma'}_{i'k'l'j'}(\mathbf{k}, i\nu_n)\, W^{s\sigma;q\sigma'}_{k'jkl'}(\mathbf{k}, i\nu_n)
$$

Formally, this represents an infinite resummation of bubble diagrams where the interaction line is repeatedly dressed by polarization insertions. The matrix equation must be solved for each momentum $\mathbf{k}$ and bosonic frequency $i\nu_n$.

The `WLat` class (child of `BLatDyn`) constructs $W$ from $P$ and $V$.

## GW Correlation Self-Energy

The *GW* correlation self-energy captures dynamical correlations through the frequency-dependent screened interaction. The correlation part is given by:

$$
\Sigma^{C,GW\, pq}_{ij\sigma}(\mathbf{k}, i\omega_n) = -\int_0^\beta d\tau \sum_{\mathbf{R}} \sum_{kl} G^{qp}_{lk\sigma}(\mathbf{R}, \tau) \times W^{C\, p\sigma q\sigma'}_{ijkl}(\mathbf{R}, \tau) \delta_{\sigma\sigma'} e^{i(\mathbf{k}\cdot\mathbf{R} - \omega_n \tau)}
$$

where $W^C = W - V$ is the dynamical part of the screened interaction. The subtraction isolates the dynamical screening contribution, avoiding double-counting since static Coulomb interaction effects are already incorporated through $\Sigma^F$.

The `SigmaGWC` class (child of `FLatDyn`) computes this self-energy by convolving the Green's function $G$ with the correlation part of the screened interaction $W^C$.

## Self-Consistent GW Loop

The full *GW* self-consistent cycle proceeds as:

1. **Non-interacting setup**: Construct $H_0$ (`NIHamiltonian`) and $V$ (`VBare`). Build $G_0$ (`GreenBare`).

2. **Initialize Green's function**: Set $G = G_0$ for the first iteration, or use the previous iteration's $G$.

3. **Polarizability**: Compute $P = GG$ (`PolLat`).

4. **Screened interaction**: Solve the bosonic Dyson equation $W = V + VPW$ (`WLat`).

5. **Correlation self-energy**: Compute $\Sigma^{C,GW} = -GW^C$ (`SigmaGWC`).

6. **Static self-energies**: Compute $\Sigma^H$ (`SigmaHartree`) and $\Sigma^F$ (`SigmaFock`) from the updated density.

7. **Dyson equation**: Combine all self-energy contributions via the Dyson equation to obtain the updated $G$ (`GreenInt`): $\Sigma = \Sigma^H + \Sigma^F + \Sigma^{GW}$, then $G = G_0 + G_0 \Sigma G$.

8. **Density and chemical potential**: Extract $n = -G(\tau = \beta^-)$ and adjust $\mu$ to maintain the target filling.

9. **Convergence check**: If converged, proceed to post-processing; otherwise, return to step 3 with the new $G$.

## Post-Processing: Quasiparticle Properties

After the *GW* loop converges, two additional classes extract quasiparticle properties from the frequency-dependent self-energy:

### Renormalization Factor (ZFactor)

The `ZFactor` class extracts the quasiparticle renormalization factor from the converged *GW* self-energy. It receives $\Sigma(\mathbf{k}, i\omega_n)$ stored in DLR representation and computes the inverse renormalization factor:

$$
Z(\mathbf{k})^{-1} = 1 - \frac{\partial \Sigma(\mathbf{k}, i\omega_n)}{\partial i\omega_n} \bigg|_{\omega=0}
$$

The $\mathbf{k}$-resolved renormalization factor $Z(\mathbf{k})$ encodes the dynamical mass enhancement and spectral weight transfer from coherent quasiparticle peaks to incoherent satellite structures.

### Static Self-Energy (SigmaStc)

The `SigmaStc` class computes the static limit of the self-energy $\Sigma(\mathbf{k}, \omega=0)$ from the full frequency-dependent *GW* result. It receives $\Sigma(\mathbf{k}, i\omega_n)$ in DLR representation and evaluates its zero-frequency limit. This static self-energy captures the shifts in quasiparticle energies.

### Quasiparticle Hamiltonian

Together, these quantities define the quasiparticle Hamiltonian:

$$
H_{QP}(\mathbf{k}) = \sqrt{Z(\mathbf{k})} \left( H_0(\mathbf{k}) + \Sigma(\mathbf{k}, \omega=0) \right) \sqrt{Z(\mathbf{k})}
$$

where $H_0(\mathbf{k})$ is the non-interacting Hamiltonian, $\Sigma(\mathbf{k}, \omega=0)$ is the static self-energy from `SigmaStc`, and $Z(\mathbf{k})$ is the renormalization factor from `ZFactor`. The symmetric placement of $\sqrt{Z}$ ensures Hermiticity when $Z(\mathbf{k})$ is a matrix in orbital space. Diagonalizing $H_{QP}(\mathbf{k})$ yields the renormalized quasiparticle band structure, incorporating both the static self-energy shift and the dynamical mass enhancement.

## References

- L. Hedin, *Phys. Rev.* **139**, A796 (1965).
- M. S. Hybertsen & S. G. Louie, *Phys. Rev. B* **34**, 5390 (1986).
- R. Godby, M. Schluter & L. Sham, *Phys. Rev. B* **37**, 10159 (1988).
- S. Massidda *et al.*, *Phys. Rev. Lett.* **74**, 2323 (1995).
- W. G. Aulbur, L. Jonsson & J. W. Wilkins, in *Solid State Physics* Vol. 54 (Academic Press, 2000).
- P. Fulde, in *Semiconductors and Insulators* (Springer, 1995).
- J. Kaye, K. Chen & H. U. Strand, *Comput. Phys. Commun.* **280**, 108458 (2022).
- J. Kaye, K. Chen & O. Parcollet, *Phys. Rev. B* **105**, 235115 (2022).
