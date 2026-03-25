# Green's Function Formalism

## Definition

The imaginary-time single-particle Green's function at finite temperature $T = 1/(k_B\beta)$ is defined as:

$$
G^{p,q}_{ij\sigma}(\mathbf{R}-\mathbf{R}', \tau - \tau') = -\langle T_\tau\, c^p_{i\sigma}(\mathbf{R}, \tau)\, c^{\dagger\, q}_{j\sigma}(\mathbf{R}', \tau') \rangle
$$

where $T_\tau$ is the imaginary-time ordering operator, $c^p_{i\sigma}$ annihilates an electron at site $(\mathbf{R}, p)$ with orbital $i$ and spin $\sigma$, and $\langle \cdots \rangle$ denotes the grand-canonical thermal average. The indices $p, q$ label basis sites and $i, j$ label orbitals.

## (Anti-)Periodicity

For $0 < \tau < \beta$, the Green's function satisfies:

$$
G(-\tau) = \xi\, G(\beta - \tau), \qquad \xi = \begin{cases} -1 & \text{(fermion)} \\ +1 & \text{(boson)} \end{cases}
$$

This boundary condition restricts the Matsubara frequencies to odd (fermionic) or even (bosonic) values.

## Matsubara Frequency Representation

The Fourier transform to Matsubara frequency is:

$$
G^{pq}_{ij\sigma}(\mathbf{k}, i\omega_n) = \frac{1}{N_\mathbf{k}} \int_0^\beta d(\tau-\tau') \sum_{\mathbf{R},\mathbf{R}'} G^{pq}_{ij\sigma}(\mathbf{R}-\mathbf{R}', \tau-\tau')\, e^{i(\mathbf{k}\cdot(\mathbf{R}-\mathbf{R}') - \omega_n \tau)}
$$

with Matsubara frequencies:

$$
i\omega_n = \begin{cases} i(2n+1)\pi/\beta & \text{(fermion)} \\ i\,2n\pi/\beta & \text{(boson)} \end{cases}
$$

## Lehmann (Spectral) Representation

The Green's function admits a spectral representation:

$$
G(\tau) = -\int_{-\infty}^{\infty} \frac{e^{-\omega\tau}}{1 + \xi\, e^{-\beta\omega}}\,\rho(\omega)\,d\omega
$$

where $\rho(\omega)$ is the spectral function. In Matsubara frequency this becomes:

$$
G(i\omega_n) = \int_{-\infty}^{\infty} \frac{\rho(\omega)}{i\omega_n - \omega}\,d\omega
$$

The spectral function connects directly to experiment: $A(\omega) = -\frac{1}{\pi}\mathrm{Im}\,G^R(\omega)$, where $G^R$ is the retarded Green's function obtained by analytic continuation $i\omega_n \to \omega + i0^+$.

## Dyson Equation

The interacting Green's function $G$ is related to the non-interacting $G_0$ through the Dyson equation:

$$
G^{pq}_{ij\sigma}(\mathbf{k}, i\omega_n) = G^{pq}_{0\,ij\sigma}(\mathbf{k}, i\omega_n) + \sum_{r,s}\sum_{k,l} G^{pr}_{0\,ik\sigma}(\mathbf{k}, i\omega_n)\,\Sigma^{rs}_{kl\sigma}(\mathbf{k}, i\omega_n)\,G^{sq}_{lj\sigma}(\mathbf{k}, i\omega_n)
$$

where $\Sigma$ is the irreducible self-energy. In QAssemble, the self-energy includes Hartree ($\Sigma^H$), Fock ($\Sigma^F$), and optionally GW correlation ($\Sigma^{C,GW}$) contributions depending on the level of theory.

## Lattice Fourier Transform

For periodic systems, transformations between momentum $\mathbf{k}$ and real space $\mathbf{R}$ are:

$$
G^{pq}_{ij\sigma}(\mathbf{k}) = \sum_{\mathbf{R}} G^{pq}_{ij\sigma}(\mathbf{R})\, e^{-i\mathbf{k}\cdot(\mathbf{R}+\tau_p-\tau_q)}, \qquad
G^{pq}_{ij\sigma}(\mathbf{R}) = \frac{1}{N_\mathbf{k}} \sum_{\mathbf{k}} G^{pq}_{ij\sigma}(\mathbf{k})\, e^{i\mathbf{k}\cdot(\mathbf{R}+\tau_p-\tau_q)}
$$

where $\tau_p$, $\tau_q$ are basis vectors. These transforms apply equally to Green's functions, self-energies, and interaction matrices.

## QAssemble Implementation

| Class | Description |
|---|---|
| `GreenBare` | Non-interacting $G_0$ from $H_0$: $G_0(i\omega_n) = (i\omega_n - H_0)^{-1}$ |
| `GreenInt` | Interacting $G$ via Dyson equation; manages $\mu$ and density $n = -G(\tau = \beta^-)$ |
| `FLatDyn` | Base class for dynamic (frequency-dependent) fermionic lattice quantities |
| `BLatDyn` | Base class for dynamic bosonic lattice quantities |

## References

- A. A. Abrikosov, L. P. Gorkov & I. E. Dzyaloshinski, *Methods of Quantum Field Theory in Statistical Physics* (Dover, 1975).
- A. L. Fetter & J. D. Walecka, *Quantum Theory of Many-Particle Systems* (Dover, 2003).
- G. D. Mahan, *Many-Particle Physics*, 3rd ed. (Springer, 2000).
