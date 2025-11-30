import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import time, datetime
qapath = os.environ.get("QAssemble")
sys.path.append(qapath + "/src")
from qacore.BLatDyn import PolLat, WLat, WLat_k

# from qacore.FPathDyn import *
# from qacore.CorrelationFunction import CorrelationFunction
# from qacore.Crystal import Crystal
from qacore.CorrelationFunction import CorrelationFunction
from qacore.Crystal import Crystal
from qacore.BLatDyn import PolLat,WLat,WLat_k
from qacore.FLatDyn import SigmaGWC
from qacore.FLatStc import (
    FLatStc,
    Hamiltonian,
    NIHamiltonian,
    SigmaFock,
    SigmaHartree,
    SigmaHartree_k,
)

filename = "glob.h5"
if os.path.exists(filename):
    os.remove(filename)
filename = "test.h5"
if os.path.exists(filename):
    os.remove(filename)



RVec = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
Basis = [[[1 / 2, 1 / 2, 1 / 2], 3]]
# Basis = [[[1 / 2, 1 / 2, 1 / 2], 3],[[0, 0, 0], 2]]
NSpin = 1
SOC = False
# KGrid = [7, 7, 7]
KGrid = [3, 3, 3]
NElec = 3 # 1 ## should be 1 for validation
beta = 100
T = 2000
cutoff = 30 #100 #30
cry = {
    "RVec": RVec,
    "Basis": Basis,
    "CorF": "F",
    "SOC": SOC,
    "NSpin": NSpin,
    "NElec": NElec,
    "KGrid": KGrid,
}
ft = {"beta": beta, "cutoff": cutoff}
# ft = {"T": T, "cutoff": cutoff}
cf = CorrelationFunction(cry=cry, ft=ft)
# print(cf.crystal.bbasis)
# print(cf.crystal.pbasis)
# exit()
t = 1.0
hopping = {
    ((0, 0), (0, 0)): {
        t: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 1), (0, 1)): {
        t: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 2), (0, 2)): {
        t: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
}
cf.TightBinding(hopping=hopping, fn=None)


from qacore.FPathStc import FPathStc

fpathstc = FPathStc(crystal=cf.crystal, obj=cf.niham)


# fpathstc.Dos(matr=cf.niham.r,plotoption=True)


kpath = [[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]]
nkpath = 101
fpathstc.crystal.Kpath(kpath, nkpath)





U = 5 #1
J = 0.1
Up = U - 2* J
V = 0.2 # 1.0
# U = 5
# Up = 0
# J = 0
# V = 0.0 # 1.0

locoption = {
    "Parameter": "SlaterKanamori",
    "option": {1: {"l": 2, "value": [U, Up, J], "orbitals": [0, 1, 2]}},
}
nonlocoption = {
    ((0, 0), (0, 0)): {
        V: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 0), (0, 1)): {
        V: [[0,0,0],[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 0), (0, 2)): {
        V: [[0,0,0],[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 1), (0, 1)): {
        V: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 1), (0, 2)): {
        V: [[0,0,0],[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
    ((0, 2), (0, 2)): {
        V: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    },
}


from qacore.BLatStc import VBare

vbare = VBare(
    crystal=cf.crystal,
    orboption=locoption,
    intamp=nonlocoption,
    hdf5file="test.h5",
    group="gw",
)


itermax = 1
mix = 0.1

# print(cf.crystal.ns)

# exit()

# ndim = 6
# norbc = 3 # self.crystal.fprojector.shape[1]
# ns = 1 # self.crystal.ns
# nspin = 2

# print(norbc,ns)

# nft = len(obsjson["occupation-susceptibility-bulla"]['0_0']['function'])

# tempmat = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nft), dtype=np.complex128, order='F')

# for ind1 in range(ndim):
#     nn1 = [0]*2
#     ind1, [iorb, ispin] = cf.crystal.indexing(ndim,2,[norbc,nspin],0,ind1,nn1)
#     for ind2 in range(ndim):
#         nn2 = [0]*2
#         ind2, [jorb, jspin] = cf.crystal.indexing(ndim,2,[norbc,nspin],0,ind2,nn2)
#         name = str(ind1)+'_'+str(ind2)
#         print(ind1,ind2,'------',iorb,ispin,jorb,jspin)

# exit()

# print(len(cf.ft.omega),len(cf.ft.nu),len(cf.ft.tau))

# exit()



cf.GWApproximation(
    itermax=itermax,
    mix=mix,
    hoppinglist=hopping,
    loccoulomb=locoption,
    nonloccoulomb=nonlocoption,
)


# fpathstc.Band(hmat = cf.niham.r+cf.sigmah.r+cf.sigmaf.r, plotoption=True)

# exit()














##########################################################################
############### Local Double-Counting part ###############################
##########################################################################


print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Start computing Local Double-CounGreenLocting GW")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

from qacore.BLocDyn import BLocDyn, PolLGW, WLoc,WLoc_temp
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

gint = cf.green
cf.crystal.Projector(impdict={"1": [[[0, 0],[0, 1],[0, 2]]]})

import pprint

pprint.pprint(cf.crystal.bimpdict)
print(cf.crystal.bimpdict['1'][0])
print(len(cf.crystal.bimpdict['1'][0]))



##############
#### Gloc ####
##############
print("**GreenLoc start")
gloc = GreenLoc(crystal=cf.crystal, ft=cf.ft, green=gint)
print("**GreenLoc finish\n")

gloc.Occ()
gint.Occ()


##############
#### Vloc ####
##############
print("**Vloc start")
# vloc = cf.vbare.Projection(cf.vbare.k)
vloc2 = np.zeros_like(cf.vbare.k)
(norb, norb, ns, ns, nk) = cf.vbare.k.shape
nspace = cf.crystal.fprojector.shape[3]
vloc = np.zeros((norb, norb, ns, ns, nspace), dtype=np.complex128, order="F")
for ik in range(nk):
    vloc2[..., ik] = cf.vbare.vloc.vloc
# vloc2[...,0] = cf.vbare.vloc.vloc
vloc[..., 0] = vloc2[..., 0]
print("**Vloc finish\n")


################
#### Polloc ####
################
print("**PolLoc start")
polloc_dc = PolLGW(crystal=cf.crystal, ft=cf.ft, green=gloc)
# pollat = PolLat(crystal=cf.crystal, ft=cf.ft, green=g_average2)
print("**PolLoc finish\n")


##############
#### Wloc ####
##############
print("**WLoc start")
start = time.time()
wloc = WLoc_temp(crystal=cf.crystal, ft=cf.ft, pol=polloc_dc.rf, vLoc=vloc)
# wloc = WLoc(crystal=cf.crystal, ft=cf.ft, wlat=cf.w)
end = time.time()
tiem_delta = end-start
print(round(tiem_delta,5))
print("**WLoc finish\n")







##########################################################################
###############         EDMFT part         ###############################
##########################################################################

print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nStart computing Local DMFT CTQMC input -- Eimp, Delta, Ubar\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


from qacore.BLocDyn import BLocDyn, PolLGW, WLoc
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

from qacore.FLocStc import EImp
from qacore.FLocDyn import Hybridisation,FWeiss
from qacore.BLocDyn import BWeiss



imp={'temperature'            : 300, # temperature (in K)
     '1':
     {
      'impurity_matrix': [ # equivalent orbital index matrix. starting from 1.
         [1,0,0],
         [0,1,0],
         [0,0,1]
         ],       
     'thermalization_time': 3,
     'measurement_time': 20,
     'green_cutoff':  40,  
     'coulomb': 'full',
     }}

# equiv = np.array([[1,2,0],
#                   [2,1,0],
#                   [0,0,1]])

equiv = cf.crystal.read_imp_equi_mat(imp)

iter = 1
key = 1

print('number of problems -- ',len(cf.crystal.probspace))

# exit()

##############
#### Uimp ####
##############
print("**Uimp start")
norb,_,ns,_,nft,nprob=wloc.crf.shape
wloc_rf_temp = np.zeros((norb,norb,ns,ns,nft,nprob))
for ift in range(nft):
    wloc_rf_temp[:,:,:,:,ift,:] = wloc.crf[:,:,:,:,ift,:] + vloc[...]

uimp = BWeiss(crystal=cf.crystal,ft=cf.ft,wloc=wloc_rf_temp,ploc=polloc_dc.rf,vloc=vloc)
print("**Uimp finish\n")



print("**SigmaLGWC start")
sigma_loc_dc = SigmaLGWC(crystal=cf.crystal, ft=cf.ft, green=gloc.gt, wloc=wloc.crt)
print("**SigmaLGWC finish\n")


print("**SigmaHLoc start")
hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=uimp.utilde_rf)
# hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=vloc)
print("**SigmaHLoc finish\n")


print("**SigmaFLoc start")
floc = SigmaFLoc(crystal=cf.crystal, ft=cf.ft, occr=gloc.occ, vloc=vloc)
print("**SigmaFLoc finish\n")







##############
#### Gimp ####
##############
print("***Weiss Green's function start")
weiss_green = FWeiss(crystal=cf.crystal,ft=cf.ft,niham=cf.niham.k,mu=cf.green.mu,hamh=cf.sigmah.k,hamf=cf.sigmaf.k,hloc=hloc.r,floc=floc.r,gloc=gloc.gf,sigmahimp=hloc.r,sigmafimp=floc.r,sigmacimp=sigma_loc_dc.rf)
print("***Weiss Green's function finish")







#####################
#### CTQMC start ####
#####################
print('\n*** write_ctqmc_params start ***')
# delta.write_ctqmc_params(1,1,eimp,equiv,vloc)
cf.CTQMCPreProcessing(iter=iter, key=key, E_imp=weiss_green.Eimp_r, imp=imp, equiv=equiv, vloc=vloc, Hyb=weiss_green.delta_rf, bweiss=uimp.utilde_rf)
print('*** write_ctqmc_params finish ***')

print('\n*** run and measure CTQMC start ***')
# cf.CTQMCRun()
# cf.CTQMCMeasure()
print('*** run and measure CTQMC finish ***')


print('\n*** impurity postprocessing start ***')
(sigmah_edmft, 
 sigmaf_edmft, 
 sigmac_edmft, 
 Chi_edmft_4) = cf.CTQMCPostProcessing(iter=iter,key=key,equiv=equiv,utilde_rf=uimp.utilde_rf)
print('*** impurity postprocessing finish ***')

#################################################
##### a class for Sigma_hfc_edmft is needed #####
#################################################



print(sigmah_edmft.r.shape)
print(sigmaf_edmft.r.shape)
print(sigmac_edmft.rf.shape)
print(Chi_edmft_4.shape)

print('===')

print(sigmah_edmft.r)
print(sigmaf_edmft.r)

# exit()





# plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], green_edmft_freq[0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
# plt.title("Green's function")
# # plt.xlabel("freq")
# # plt.ylabel("F_Loc")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()


# plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], sigmac_edmft_freq[0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
# plt.title("Sigma_C")
# # plt.xlabel("freq")
# # plt.ylabel("F_Loc")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()



# plot = plt.figure(1)
# plt.scatter(cf.ft.nu[:], Chi_edmft_4[0, 0, 0, 0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
# plt.title("Chi")
# # plt.xlabel("freq")
# # plt.ylabel("F_Loc")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()




##########################################
##### a class for Pi_edmft is needed #####
##########################################

print("\n*** Pi_edmft start ***")

norbc,_,_,_,nspin,_,nft = Chi_edmft_4.shape
Chi_edmft_temp = np.zeros((norbc*norbc,norbc*norbc,nspin,nspin,nft),dtype=np.complex64,order='F')

for iis in range(ns):
    for jjs in range(ns):
        for ift in range(nft):
            Chi_edmft_temp[:,:,iis,jjs,ift] = cf.crystal.Quad2Double(Chi_edmft_4[:,:,:,:,iis,jjs,ift])

Chi_edmft_2 = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nft),dtype=np.complex64,order='F')

for ift in range(nft):
    Chi_edmft_2[...,0,0,ift] = 1.0/2.0*(Chi_edmft_temp[...,0,0,ift]+Chi_edmft_temp[...,0,1,ift]+Chi_edmft_temp[...,1,0,ift]+Chi_edmft_temp[...,1,1,ift])


Pi_test = uimp.Dyson(Chi_edmft_2, -uimp.utilde_rf[...,0])


plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], Pi_test[0, 0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
plt.title("Chi")
# plt.xlabel("freq")
# plt.ylabel("F_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

print("*** Pi_edmft finish ***")

######################
#### CTQMC finish ####
######################




