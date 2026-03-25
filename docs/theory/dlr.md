# Discrete Lehmann Representation (DLR)

## Lehmann Representation

Any imaginary-time Green's function admits a spectral (Lehmann) representation:

$$
G(\tau) = -\int_{-\infty}^{\infty} K(\tau, \omega)\,\rho(\omega)\,d\omega, \qquad K(\tau, \omega) = \frac{e^{-\omega\tau}}{1 + \xi\, e^{-\beta\omega}}
$$

where $\rho(\omega)$ is the spectral density and $\xi = -1$ (fermion) or $+1$ (boson). When $\rho$ is supported within $[-\omega_\mathrm{max}, \omega_\mathrm{max}]$, we define the dimensionless cutoff $\Lambda \equiv \beta\omega_\mathrm{max}$.

## DLR Approximation

The kernel $K(\tau, \omega)$ is numerically low-rank. The DLR exploits this by approximating $G(\tau)$ as a sum of $r$ exponentials:

$$
G_\mathrm{DLR}(\tau) = \sum_{l=1}^{r} \widehat{g}_l\, K(\tau, \omega_l), \qquad r = O\!\bigl(\log(\Lambda)\log(1/\epsilon)\bigr)
$$

The DLR frequencies $\{\omega_l\}$ are selected via interpolative decomposition (pivoted QR) of $K$ discretized on a composite Chebyshev grid. The same process yields $r$ imaginary-time nodes $\{\tau_k\}$ and $r$ Matsubara frequency nodes $\{i\nu_{n_k}\}$. The basis is universal: for given $\Lambda$ and $\epsilon$, the same $\{\omega_l\}$ represent any Green's function within that cutoff.

The DLR coefficients $\widehat{g}_l$ are recovered by solving an $r \times r$ interpolation problem from samples at the DLR nodes. The expansion transforms analytically to Matsubara frequency:

$$
G(i\nu_n) = \sum_{l=1}^{r} \frac{\widehat{g}_l}{i\nu_n + \omega_l}
$$

so that $\tau \leftrightarrow i\nu_n$ transforms reduce to $r \times r$ linear algebra instead of $O(\beta)$-sized FFTs.

## QAssemble Implementation

The `DLR` class constructs separate fermionic and bosonic DLR objects via `pydlr`. Key attributes and methods:

| Attribute / Method | Description |
|---|---|
| `tauF`, `tauB` | Fermionic / bosonic imaginary-time nodes |
| `omega`, `nu` | Fermionic / bosonic Matsubara frequency nodes |
| `FT2F`, `FF2T` | Fermionic $\tau \leftrightarrow i\omega_n$ transforms |
| `BT2F`, `BF2T` | Bosonic $\tau \leftrightarrow i\nu_n$ transforms |

## References

- J. Kaye, K. Chen & O. Parcollet, *Phys. Rev. B* **105**, 235115 (2022). [DOI](https://doi.org/10.1103/PhysRevB.105.235115)
- J. Kaye, K. Chen & H. U. Strand, *Comput. Phys. Commun.* **280**, 108458 (2022). [DOI](https://doi.org/10.1016/j.cpc.2022.108458)
