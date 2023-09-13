# %%
import numpy as np
import sys
path = '/home/momichael98/temp/Fortran/DiagE/class'
sys.path.append(path)
from ClassDiagE import Crystal as cr
from ClassDiagE import FHamiltonian as fh
from ClassDiagE import BHamiltonian as bh
from ClassDiagE import FT_grid as ft
from ClassDiagE import Method
path = '/home/momichael98/temp/Fortran/DiagE/modules'
sys.path.append(path)
import DiagE

# %% [markdown]
# ## Generate Crystal & Basic system

# %%
latt = [[1.0,0.0,0.0],[1/2,np.sqrt(3)/2,0.0],[0.0,0.0,10.0]]
pos = [[1/3,1/3,1/2],[2/3,2/3,1/2]]

Graphene = cr(latt,pos)
grid = [15,15,1]
Graphene.Kpoint(meshgrid=grid)

# %%
Gf = fh(Graphene,1)
Gb = bh(Graphene,1)

orb_option1 = [0,1]
orb_option2 = [1,1]

Gf.set_basis_index(orb_option1)
Gf.set_basis_index(orb_option2)

Gb.set_basis_index(orb_option1)
Gb.set_basis_index(orb_option2)

# %% [markdown]
# ## Construct Non-interacting Hamiltonian

# %%
t = -2.7
e = 0.0
Gf.On_site_list(e)
Gf.On_site_list(e)

Gf.Hoppinglist(t,0,1,0,0,0)
Gf.Hoppinglist(t,1,0,1,0,0)
Gf.Hoppinglist(t,1,0,0,1,0)

Ham_tb = Gf.Hamiltonian(Gf.kpoint)

# %% [markdown]
# ### Visualization of tight_binding Hamiltonian

# %%
energy = Gf.diagonalization(Ham_tb)
Gf.visualization(energy,'./png/Graphene_tb')

# %% [markdown]
# ## Construct the interacting Hamiltonian

# %%
U = 5
option1 = {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbital" : [0]}
option2 = {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbital" : [1]}

Gb.local_interacting(option1)
Gb.local_interacting(option2)

V = 2

Gb.set_int_amp(V,1,0,[1,0,0])
Gb.set_int_amp(V,1,0,[0,1,0])
Gb.set_int_amp(V,0,1,[0,0,0])

Gb.gen_nl_int_ham(grid)
Gb.Combine_interaction()


# %% [markdown]
# ## Hartree-Fock approximation

# %% [markdown]
# ### Construct FT grid & Method class

# %%
beta = 38.0
size = 1000
FT = ft(beta,size)

FT.Omega()
FT.Tau()
FT.Nu()

GM = Method(Gf,Gb,FT)

# %% [markdown]
# ### SCF loop of Hartree-Fock

# %%
iter = 100
Nt = 1
Hmat_hf, Sigma_H, Sigma_F, n = GM.Self_consistence_Hartree_Fock(iter,Ham_tb,Nt)

# %%
energy_hf = Gf.diagonalization(Hmat_hf)
Gf.visualization(energy_hf,'./png/Graphene_HF')

# %%


# %% [markdown]
# ## GW approximation

# %%
G_full_kf, Sigma_H, Sigma_F, Sigma_C_kf, Pol_kf, Wc_kf, mu = GM.Self_consistence_GW(iter,Ham_tb,Nt)

# %%
Sigma_stc = Gf.Stc_Correlated_self_energy(Sigma_C_kf)


# %%
Z = Gf.z_factor(Sigma_C_kf,beta)

# %%
Z.shape

# %%
H_qp = Gf.QP_Hamiltonian(Ham_tb,Sigma_H,Sigma_F,Sigma_stc,mu,Z)

# %%
energy_QP = Gf.diagonalization(H_qp)

# %%
Gf.visualization(energy_QP,'./png/Graphene_GW')
