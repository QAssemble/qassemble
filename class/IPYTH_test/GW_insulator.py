import numpy as np
import sys
path = '/home/momichael98/temp/Fortran/DiagE/class'
sys.path.append(path)
from ClassDiagE_new import Crystal as cr
from ClassDiagE_new import FHamiltonian as fh
from ClassDiagE_new import BHamiltonian as bh
from ClassDiagE_new import FT_grid as ft
from ClassDiagE_new import Method
path = '/home/momichael98/temp/Fortran/DiagE/modules'
sys.path.append(path)
import DiagE
import matplotlib
import matplotlib.pyplot as plt


latt = [[1.0,0.0,0.0],[1/2,np.sqrt(3)/2,0.0],[0.0,0.0,10.0]]
pos = [[1/3,1/3,1/2],[2/3,2/3,1/2]]

Graphene = cr(latt,pos)
grid = [15,15,1]
Graphene.Kpoint(meshgrid=grid)

Gf = fh(Graphene,1)
Gb = bh(Graphene,1)

orb_option1 = [0,1]
orb_option2 = [1,1]

Gf.set_basis_index(orb_option1)
Gf.set_basis_index(orb_option2)

Gb.set_basis_index(orb_option1)
Gb.set_basis_index(orb_option2)

t = -2.7
e = 2
Gf.On_site_list(e)
Gf.On_site_list(-e)

Gf.Hoppinglist(t,0,1,0,0,0)
Gf.Hoppinglist(t,1,0,1,0,0)
Gf.Hoppinglist(t,1,0,0,1,0)

Ham_tb = Gf.Hamiltonian(Gf.kpoint)

energy = Gf.diagonalization(Ham_tb)
Gf.visualization(energy,'./png/tight_binding.png')
print(energy[:,:,1].min() - energy[:,:,0].max())

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

beta = 38.0
size = 1000
FT = ft(beta,size)

FT.Omega()
FT.Tau()
FT.Nu()

GM = Method(Gb)
GM.mapping_full_sub()
GM.mapping_mR_R(grid)

iter = 100
mu = 10
Nt = 1
H_hf, Hartree, Fock = GM.SCF_Hartree_Fock(iter,Ham_tb,FT.tau,mu,Nt,GM.V_bare,grid)

energy_hf = Gf.diagonalization(H_hf)
Gf.visualization(energy_hf,'./png/HF.png')
print(energy_hf[:,:,1].min() - energy_hf[:,:,0].max())

G_full, Hartree, Fock, Sigma_C, Chem = GM.SCF(100,Ham_tb,GM.V_bare,grid,FT,0,Nt)
Sigma_stc = GM.Stc_Correlated_self_energy(Sigma_C)
Z = GM.z_factor(Sigma_C,beta)
H_qp = GM.QP_Hamiltonian(Ham_tb,Hartree,Fock,Sigma_stc,Chem[:,:,:,:,0],Z)

energy_QP = Gf.diagonalization(H_qp)
Gf.visualization(energy_QP,'./png/GW.png')
print(energy_QP[:,:,1].min() - energy_QP[:,:,0].max())
