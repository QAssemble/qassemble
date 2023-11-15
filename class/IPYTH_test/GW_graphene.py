import numpy as np
import sys,os,time,datetime
diage_path = os.environ.get('DIAGE','')

path = diage_path+'/class'
sys.path.append(path)
from ClassDiagE import Crystal as cr
from ClassDiagE import FHamiltonian as fh
from ClassDiagE import BHamiltonian as bh
from ClassDiagE import FT_grid as ft
from ClassDiagE import Method
from ClassDiagE import Impurity
path = diage_path+'/modules'
sys.path.append(path)
import DiagE

latt = [[1.0,0.0,0.0],[1/2,np.sqrt(3)/2,0.0],[0.0,0.0,10.0]]
pos = [[1/3,1/3,1/2],[2/3,2/3,1/2]]

Graphene = cr(latt,pos)
grid = [10,10,1]
Graphene.Kpoint(meshgrid=grid)


Gf = fh(Graphene,1)
Gb = bh(Graphene,Gf)

orb_option1 = [0,1]
orb_option2 = [1,1]

Gf.set_basis_index(orb_option1)
Gf.set_basis_index(orb_option2)

Gb.set_basis_index(orb_option1)
Gb.set_basis_index(orb_option2)

t = -2.7
e = 0.0
Gf.On_site_list([e,e])

Gf.Hoppinglist(t,0,1,[0,0,0])
Gf.Hoppinglist(t,1,0,[1,0,0])
Gf.Hoppinglist(t,1,0,[0,1,0])
Gf.Hamiltonian()
Ham_tb = Gf.Ham_tb

# energy = Gf.diagonalize(Ham_tb)
# Gf.visualization(energy)

U = 5
option1 = {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbital" : [0]}
option2 = {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbital" : [1]}

Gb.local_interacting(option1)
Gb.local_interacting(option2)

V = 2

Gb.set_int_amp(V,1,0,[1,0,0])
Gb.set_int_amp(V,1,0,[0,1,0])
Gb.set_int_amp(V,0,1,[0,0,0])

Gb.gen_nl_int_ham()
Gb.Combine_interaction()

beta = 38.0
size = 1000
FT = ft(beta,size)

FT.Omega()
FT.Tau()
FT.Nu()


iter = 100
Nt = 1
mix = 0.05
Gi = Impurity(Gf,Gb,FT)
ind_dict = {"1":[[[0,0]],[[1,0]]]}
Gi.projector(ind_dict)

GM = Method(Gf,Gb,FT,Gi)
# print('Hartree-Fock start')
# Hmat_hf, Sigma_H, Sigma_F, n, mu_hf = GM.Hartree_Fock(iter,Ham_tb,Nt,mix)
# print('Hartree-Fock finish')

# energy_hf = Gf.diagonalize(Hmat_hf)
# Gf.visualization(energy_hf,'./png/Hartre-Fock.png')
# print("GW start")
# G_full_kf, Sigma_H, Sigam_F, Sigma_C_kf, Sigma, Pol_kf, Wc_kf, mu = GM.GW_approximation(iter,Gf.Ham_tb,Nt,mix)
# print("GW finish")

# Sigma_stc = Gf.Stc_self_energy(Sigma)

# Z = Gf.z_factor(Sigma,beta)

# H_qp = Gf.QP_Hamiltonian(Gf.Ham_tb,-Sigma_H,-Sigma_F,Sigma_stc,mu,Z)

# energy3 = Gf.diagonalize(H_qp)
# Gf.visualization(energy3,'./png/GW-approximation.png')

equiv = np.array([[1]])
print("DMFT start")
start = time.time()
G_latfreq,Sigma_imp,E_imp,hyb,mu,Sigma_emb = GM.DMFT(int(iter/100),Ham_tb,Nt,mix,None,0,0,equiv)
end = time.time()
sec = (end-start)
delta = datetime.timedelta(seconds=sec)
print("DMFT finish")
print(f"DMFT loop time : {delta}")

# A = -1/np.pi*np.imag(G_latfreq)
# print(A.shape)