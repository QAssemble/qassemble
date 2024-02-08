
# Table of Contents

1.  [Model Hamiltonian: Extended Hund Hubbard model](#org4fd2f8c)
    1.  [Hamiltonian in [1] S. Ryee, P. Sémon, M. J. Han, and S. Choi, Nonlocal Coulomb Interaction and Spin-Freezing Crossover as a Route to Valence-Skipping Charge Order, Npj Quantum Mater. 5, 1 (2020](#org3735754)
    2.  [3D not in 2D](#org7f124a9)
2.  [Simulation condition](#orgfe96dd1)
    1.  [Two electron filling](#org7452b78)
    2.  [t=1 eV](#orgce2df72)
    3.  [U=1eV](#org95d7098)
    4.  [J=0.1eV](#orgf28b502)
    5.  [V=0.2eV](#org3a09ccf)
    6.  [T=100K](#org6a8a3f0)
    7.  [10\*10\*10 k mesh](#org5bf3119)


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

## T=100K


<a id="org5bf3119"></a>

## 10\*10\*10 k mesh

