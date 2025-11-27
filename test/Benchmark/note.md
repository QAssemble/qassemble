
# Table of Contents

- [Table of Contents](#table-of-contents)
- [Model Hamiltonian: Extended Hund Hubbard model](#model-hamiltonian-extended-hund-hubbard-model)
  - [Hamiltonian in \[1\] S. Ryee, P. Sémon, M. J. Han, and S. Choi, Nonlocal Coulomb Interaction and Spin-Freezing Crossover as a Route to Valence-Skipping Charge Order, Npj Quantum Mater. 5, 1 (2020](#hamiltonian-in-1-s-ryee-p-sémon-m-j-han-and-s-choi-nonlocal-coulomb-interaction-and-spin-freezing-crossover-as-a-route-to-valence-skipping-charge-order-npj-quantum-mater-5-1-2020)
  - [3D not in 2D](#3d-not-in-2d)
- [Simulation condition](#simulation-condition)
  - [Two electron filling](#two-electron-filling)
  - [t=1 eV](#t1-ev)
  - [U=1eV](#u1ev)
  - [J=0.1eV](#j01ev)
  - [V=0.2eV](#v02ev)
  - [T=100K](#t100k)
  - [10\*10\*10 k mesh](#101010-k-mesh)


<a id="org4fd2f8c"></a>

# Model Hamiltonian: Extended Hund Hubbard model


<a id="org3735754"></a>

## Hamiltonian in [1] S. Ryee, P. Sémon, M. J. Han, and S. Choi, Nonlocal Coulomb Interaction and Spin-Freezing Crossover as a Route to Valence-Skipping Charge Order, Npj Quantum Mater. 5, 1 (2020


<a id="org7f124a9"></a>

## 3D not in 2D

$$\mathcal{H} =& -t\sum_{\langle ij \rangle,\gamma,\sigma}{\big(c^{\dagger}_{i\gamma \sigma}c_{j\gamma \sigma} + \mathrm{H.c.}\big)} - \mu\sum_{i,\gamma,\sigma}{n_{i \gamma \sigma}} \nonumber \\ 
&+ H_{\mathrm{loc}} + H_{\mathrm{nonloc}}$$
$$H_\mathrm{loc} &= U\sum_{i,\gamma,\sigma}{n_{i \gamma \uparrow} n_{i \gamma \downarrow}}+ (U-2J)\sum_{i,\gamma,\gamma'}^{\gamma \neq \gamma'}{ n_{i \gamma \uparrow} n_{i \gamma' \downarrow}} \nonumber + (U-3J)\sum_{i,\gamma,\gamma',\sigma}^{\gamma < \gamma'}{ n_{i \gamma \sigma} n_{i \gamma' \sigma}}  \nonumber - J\sum_{i,\gamma,\gamma'}^{\gamma \neq \gamma'}{(c^{\dagger}_{i \gamma \uparrow} c_{i \gamma \downarrow} c^{\dagger}_{i \gamma' \downarrow} c_{i \gamma' \uparrow} + c^{\dagger}_{i \gamma \uparrow} c^{\dagger}_{i \gamma \downarrow} c_{i \gamma' \uparrow} c_{i \gamma' \downarrow})}.$$

$$H_\mathrm{nonloc} = \sum_{\substack{\langle ij \rangle \\ \gamma,\gamma',\sigma,\sigma'}}{Vn_{i \gamma \sigma}n_{j \gamma' \sigma'}}$$


<a id="orgfe96dd1"></a>

# Simulation condition


<a id="org7452b78"></a>

## Two electron filling


<a id="orgce2df72"></a>

## t=1 eV


<a id="org95d7098"></a>

## U=1eV


<a id="orgf28b502"></a>

## J=0.1eV


<a id="org3a09ccf"></a>

## V=0.2eV


<a id="org6a8a3f0"></a>

## beta=100K


<a id="org5bf3119"></a>

## 10\*10\*10 k mesh

