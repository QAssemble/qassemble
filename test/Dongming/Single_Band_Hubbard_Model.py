import os
import sys

import matplotlib.pyplot as plt
import numpy as np

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
Basis = [[[1 / 2, 1 / 2, 1 / 2], 1]]
NSpin = 1
SOC = False
KGrid = [9, 9, 9]
NElec = 1
T = 2000
cutoff = 100
cry = {
    "RVec": RVec,
    "Basis": Basis,
    "CorF": "F",
    "SOC": SOC,
    "NSpin": NSpin,
    "NElec": NElec,
    "KGrid": KGrid,
}
ft = {"T": T, "cutoff": cutoff}
cf = CorrelationFunction(cry=cry, ft=ft)


t = 1.0
hopping = {((0, 0), (0, 0)): {t: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}}
cf.TightBinding(hopping=hopping, fn=None)


from qacore.FPathStc import FPathStc

fpathstc = FPathStc(crystal=cf.crystal, obj=cf.niham)


# fpathstc.Dos(matr=cf.niham.r,plotoption=True)


kpath = [[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]]
nkpath = 101
fpathstc.crystal.Kpath(kpath, nkpath)


# fpathstc.Band(hmat = cf.niham.r, plotoption=True)


U = 5
Up = 0
J = 0
V = 0.0 # 1.0

locoption = {
    "Parameter": "SlaterKanamori",
    "option": {1: {"l": 0, "value": [U, Up, J], "orbitals": [0]}},
}
nonlocoption = {((0, 0), (0, 0)): {V: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}}


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


cf.GWApproximation(
    itermax=itermax,
    mix=mix,
    hoppinglist=hopping,
    loccoulomb=locoption,
    nonloccoulomb=nonlocoption,
)

















print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Start computing Local Double-CounGreenLocting GW")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

from qacore.BLocDyn import BLocDyn, PolLoc, WLoc,WLoc_temp
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

gint = cf.green
cf.crystal.Projector(impdict={"1": [[[0, 0]]]})
# cf.crystal.Projector(impdict = {"1" : [[[0, 1]]], "2" : [[[1, 2], [1,3]]], "3" : [[[2, 2]]]})
# exit()

# vloc = cf.vbare.vloc

print("**GreenLoc start")
gloc = GreenLoc(crystal=cf.crystal, ft=cf.ft, green=gint)
print("**GreenLoc finish\n")

(norb, norb, ns, nk, nf) = gint.kf.shape
g_average = np.zeros((norb, norb, ns, nk, nf), dtype=np.complex128, order="F")

for ik in range(nk):
    for i in range(len(cf.crystal.kpoint)):
        g_average[..., ik, :] += 1 / nk * gint.kf[..., i, :]

g_average2 = cf.green.F2T(g_average, 1, 1)


for iff in range(nf):
    for js in range(ns):
        for jorb in range(1):
            for iorb in range(1):
                err = (
                    g_average[iorb, jorb, js, 0, iff] - gloc.gf[iorb, jorb, js, iff, 0]
                )
                if abs(err) > 1.0e-6:
                    print(
                        iorb,
                        jorb,
                        js,
                        iff,
                        abs(err),
                        g_average[iorb, jorb, js, 0, iff],
                        gloc.gf[iorb, jorb, js, iff, 0],
                    )

plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], g_average[0, 0, 0, 0, :], color="blue")
plt.scatter(cf.ft.omega[:], gloc.gf[0, 0, 0, :, 0], color="red")
plt.title("Green_Loc")
plt.xlabel("freq")
plt.ylabel("G")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()


gloc.Occ()
gint.Occ()
# print('gloc.occ --',gloc.occ)
# print('gloc.occ.shape --',gloc.occ.shape)
# print('gloc.NumofE() --',gloc.NumofE())


print("**Vloc start")
# vloc = cf.vbare.Projection(cf.vbare.k)
# print("vloc.shape --", vloc.shape)
# print("vbare.shape --", cf.vbare.k.shape)
vloc = np.zeros((1, 1, 1, 1, 2), dtype=np.complex128, order="F")
vloc2 = np.zeros_like(cf.vbare.k)
(norb, norb, ns, ns, nk) = cf.vbare.k.shape
for ik in range(nk):
    vloc2[..., ik] = cf.vbare.vloc.vloc
vloc[..., 0] = vloc2[0, 0, 0, 0, 0]
vloc[..., 1] = vloc2[0, 0, 0, 0, 0]
print("vloc.shape --", vloc.shape)
print(vloc[:, :, 0, 0, 0])
print("vbare.shape --", cf.vbare.k.shape)
print(vloc2[:, :, 0, 0, 0])
print("**Vloc finish\n")

# exit()


# print(cf.vbare.k.shape)
# print(vloc.shape)
# exit()


(norb, norb, ns, ns, nk) = cf.vbare.k.shape
vbare_k_average = np.zeros((norb, norb, ns, ns, nk), dtype=np.complex128, order="F")

for ikk in range(nk):
    for ik in range(nk):
        vbare_k_average[..., ikk] += 1 / nk * cf.vbare.k[..., ik]


# print(vbare_k_average[...,0:5])
# print(vloc)

# exit()

# print(gloc.occ)
# print('===')
# print(vloc)
# print('===')
# print(cf.green.occ)
# print('===')
# print(vbare_k_average)


# print(gint.occ.shape)
# gint.occ[0,1,0] = 0.0
# gint.occ[1,0,0] = 0.0

# exit()


# print("**SigmaHLoc start")
# hloc = SigmaHLoc(crystal=cf.crystal, ft=cf.ft, occ=gloc.occ, vloc=vloc)
# hlat = SigmaHartree(crystal=cf.crystal, occ=-g_average2[..., 0, -1], vbare=vloc2)
# # hlat = SigmaHartree(crystal=cf.crystal,occ=gint.occ,vbare=vbare_k_average)
# print("**SigmaHLoc finish\n")

# print(hlat.r.shape)

# (norb, norb, ns, nk) = hlat.k.shape
# hartree_average = np.zeros((norb, norb, ns, nf), dtype=np.complex128, order="F")
# for iff in range(nf):
#     # for ik in range(nk):
#     #     hartree_average[...,iff] += 1/nk*hlat.r[...,0]
#     hartree_average[..., iff] = hlat.r[..., 0]

# # print(hartree_average[0,0,0,0])
# # print(hloc.r[0,0,0,0,0])
# # print(hartree_average[0,0,0,0]-hloc.r[0,0,0,0,0])

# plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], hartree_average[0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], hloc.r[0, 0, 0, :, 0], color="red")
# plt.title("Hartree_Loc")
# plt.xlabel("freq")
# plt.ylabel("H_Loc")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()

# # exit()


# print("**SigmaFLoc start")
# floc = SigmaFLoc(crystal=cf.crystal, ft=cf.ft, occr=gloc.occ, vloc=vloc)
# flat = SigmaFock(crystal=cf.crystal, occr=gint.occr, vbare=cf.vbare.r)
# print("**SigmaFLoc finish\n")

# print(flat.r.shape)
# print(floc.r.shape)

# (norb, norb, ns, nr) = flat.r.shape
# flat_average = np.zeros((norb, norb, ns, nf), dtype=np.complex128, order="F")
# for iff in range(nf):
#     # for ir in range(nr):
#     #     flat_average[...,iff] += 1/nr * flat.r[...,ir]
#     flat_average[..., iff] = flat.r[..., 0]


# print(flat.r[0, 0, 0, 0])
# print(floc.r[0, 0, 0, 0, 0])

# plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], flat_average[0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], floc.r[0, 0, 0, :, 0], color="red")
# plt.title("Fock_Loc")
# plt.xlabel("freq")
# plt.ylabel("F_Loc")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()

# # exit()











print("**PolLoc start")
polloc = PolLoc(crystal=cf.crystal, ft=cf.ft, green=gloc)
pollat = PolLat(crystal=cf.crystal, ft=cf.ft, green=g_average2)
# polloc_average = PolLoc(crystal=cf.crystal,ft=cf.ft,green=g_average)
# print('polLoc.rt.shape --',polloc.rt.shape)
print("**PolLoc finish\n")

# (norb,norb, ns, ns, nk, nf) = cf.pol.kf.shape
# pol_average = np.zeros((norb, norb, ns, ns, nf),dtype=np.complex128, order='F')
# for i in range(nk):
#     pol_average += 1/nk*cf.pol.kf[...,i,:]

for iff in range(nf):
    for js in range(ns):
        for ks in range(ns):
            for jorb in range(1):
                for iorb in range(1):
                    # err = pol_average[iorb, jorb, js, ks, iff] - polloc.rf[iorb, jorb, js, ks, iff, 0]
                    err = (
                        pollat.rf[iorb, jorb, js, ks, 0, iff]
                        - polloc.rf[iorb, jorb, js, ks, iff, 0]
                    )
                    if abs(err) > 1.0e-6:
                        print(
                            iorb,
                            jorb,
                            js,
                            ks,
                            iff,
                            abs(err),
                            pollat.rf[iorb, jorb, js, ks, 0, iff],
                            polloc.rf[iorb, jorb, js, ks, iff, 0],
                        )

plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], pollat.rf[0, 0, 0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.nu[:],polloc_average.rf[0,0,0,0,:],color='blue')
plt.scatter(cf.ft.nu[:], polloc.rf[0, 0, 0, 0, :, 0], color="red")
plt.title("Polarizability_Loc")
plt.xlabel("freq")
plt.ylabel("P")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

# print(vloc.rf)

# print(cf.vbare.rf)


# exit()

# pollat = PolLat(crystal=cf.crystal, ft=cf.ft, green=gint.rt)

# (norb,norb, ns, ns, nk, nf) = pollat.kf.shape
# pol_average = np.zeros((norb, norb, ns, ns, nk, nf),dtype=np.complex128, order='F')
# for i in range(nk):
#     pol_average[...,0,:] += 1/nk*pollat.kf[...,i,:]

# g_average2 = cf.green.F2T(g_average,1,1)


# print(vloc.shape)
# print(vbare_k_average.shape)


print(pollat.kf[..., 0:5, 0])

print(polloc.rf[..., 0, 0])

for ik in range(nk):
    pollat.kf[0, 0, 0, 0, ik, :] = pollat.rf[0, 0, 0, 0, 0, :]


# pollat.kf[0, 1, 0, 0, :, :] = 0.0
# pollat.kf[1, 0, 0, 0, :, :] = 0.0
# pollat.kf[1, 1, 0, 0, :, :] = 0.0

# vloc2[0, 1, 0, 0, :] = 0.0
# vloc2[1, 0, 0, 0, :] = 0.0
# vloc2[1, 1, 0, 0, :] = 0.0



# print(pollat.kf.shape)
# print(cf.vbare.k.shape)

# exit()

print("**WLoc start")
wloc_temp = WLoc_temp(crystal=cf.crystal, ft=cf.ft, pol=polloc.rf, vLoc=vloc)
# wlat = WLat(crystal=cf.crystal, ft=cf.ft, pol=pollat.kf, vbare=cf.vbare)
wlat = WLat_k(crystal=cf.crystal, ft=cf.ft, pol=pollat.kf, vbare=vloc2)
# wlat = WLat_k(crystal=cf.crystal,ft=cf.ft, pol=pollat.kf,vbare=vbare_k_average)
print("**WLoc finish\n")


# (norb,norb, ns, ns, nk, nf) = cf.w.ckf.shape
# wlat_average = np.zeros((norb, norb, ns, ns, nf),dtype=np.complex128, order='F')
# for i in range(nk):
#     wlat_average += 1/nk*cf.w.ckf[...,i,:]
#     # wlat_average += 1/nk*wlat.ckf[...,i,:]

# for iff in range(nf):
#     for js in range(ns):
#         for ks in range(ns):
#             for jorb in range(1):
#                 for iorb in range(1):
#                     err = wlat_average.crf[iorb, jorb, js, ks, 0, iff] - wloc.crf[iorb, jorb, js, ks, iff, 0]
#                     if (abs(err) > 1.0e-6):
#                         print(iorb, jorb, js, ks, iff, abs(err), wlat_average.crf[iorb, jorb, js, ks, 0, iff], wloc.crf[iorb, jorb, js, ks, iff, 0])

print(wlat.crf.shape)

plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], wlat.crf[0, 0, 0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.nu[:],wlat_average[0,0,0,0,:],color='blue')
plt.scatter(cf.ft.nu[:], wloc_temp.crf[0, 0, 0, 0, :, 0], color="red")
plt.title("W_Loc")
plt.xlabel("freq")
plt.ylabel("W")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()


# print("**SigmaLGWC start")
# sigma_loc_gwc = SigmaLGWC(crystal=cf.crystal, ft=cf.ft, green=gloc.gt, wloc=wloc.crt)
# # sigma_lat_gwc = SigmaGWC(crystal=cf.crystal,ft=cf.ft,green=gint.rt,wlat=cf.w.crt)
# sigma_lat_gwc = SigmaGWC(crystal=cf.crystal, ft=cf.ft, green=gint.rt, wlat=wlat.crt)
# print("**SigmaLGWC finish\n")


# # (norb,norb, ns, nk, nf) = sigma_lat_gwc.kf.shape
# # sigmagwc_lat_average = np.zeros((norb, norb, ns, nf),dtype=np.complex128, order='F')
# # for i in range(nk):
# #     sigmagwc_lat_average += 1/nk*sigma_lat_gwc.kf[...,i,:]

# print(sigma_lat_gwc.rf.shape)

# plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], sigma_lat_gwc.rf[0, 0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:],sigmagwc_lat_average[0,0,0,:],color='blue')
# plt.scatter(cf.ft.omega[:], sigma_loc_gwc.rf[0, 0, 0, :, 0], color="red")
# plt.title("Sigma")
# plt.xlabel("freq")
# plt.ylabel("Sigma")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()





print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nStart computing Local DMFT CTQMC input -- Eimp, Delta, Ubar\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


from qacore.BLocDyn import BLocDyn, PolLoc, WLoc
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

from qacore.FLocStc import EImp
from qacore.FLocDyn import Hybridisation
from qacore.BLocDyn import UImp

print(vloc.shape)

print("**Uimp start")
uimp = UImp(crystal=cf.crystal,ft=cf.ft,wloc=wloc_temp.rf,ploc=polloc.rf,vloc=vloc)
print("**Uimp finish\n")

plot = plt.figure(1)
plt.scatter(cf.ft.nu[1:], uimp.utilde_rf[0, 0, 0, 0, 1:, 0].real, color="blue")
plt.title("ubar -- real part")
plt.xlabel("freq")
plt.ylabel("Delta")
# plt.ylim(-5, 5)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

print("**WLoc start")
wloc = WLoc(crystal=cf.crystal, ft=cf.ft, pol=polloc.rf, vLoc=vloc, vDyn=uimp.utilde_rf)
print("**WLoc finish\n")
plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], wloc.rf[0, 0, 0, 0, :, 0], color="blue")
# plt.scatter(cf.ft.nu[:],wlat_average[0,0,0,0,:],color='blue')
plt.scatter(cf.ft.nu[:], wloc_temp.rf[0, 0, 0, 0, :, 0], color="red")
plt.title("W_Loc")
plt.xlabel("freq")
plt.ylabel("W")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()
print("**SigmaLGWC start")
sigma_loc_gwc = SigmaLGWC(crystal=cf.crystal, ft=cf.ft, green=gloc.gt, wloc=wloc.crt)
print("**SigmaLGWC finish\n")

plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], sigma_lat_gwc.rf[0, 0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:],sigmagwc_lat_average[0,0,0,:],color='blue')
plt.scatter(cf.ft.omega[:], sigma_loc_gwc.rf[0, 0, 0, :, 0], color="red")
plt.title("Sigma")
plt.xlabel("freq")
plt.ylabel("Sigma")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()









print("**SigmaHLoc start")
hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=uimp.utilde_rf)
# hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=vloc)
print("**SigmaHLoc finish\n")


# print("**SigmaHLoc start")
# hloc = SigmaHLoc(crystal=cf.crystal, ft=cf.ft, occ=gloc.occ, vloc=vloc)
hlat = SigmaHartree(crystal=cf.crystal, occ=-g_average2[..., 0, -1], vbare=vloc2)
# hlat = SigmaHartree(crystal=cf.crystal,occ=gint.occ,vbare=vbare_k_average)
# print("**SigmaHLoc finish\n")

print(hlat.r.shape)

(norb, norb, ns, nk) = hlat.k.shape
hartree_average = np.zeros((norb, norb, ns, nf), dtype=np.complex128, order="F")
for iff in range(nf):
    # for ik in range(nk):
    #     hartree_average[...,iff] += 1/nk*hlat.r[...,0]
    hartree_average[..., iff] = hlat.r[..., 0]

# print(hartree_average[0,0,0,0])
# print(hloc.r[0,0,0,0,0])
# print(hartree_average[0,0,0,0]-hloc.r[0,0,0,0,0])
    
hloc_constant = np.zeros(nf)
hloc_constant[:] = hloc.r[0,0,0,0]

plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], hartree_average[0, 0, 0, :], color="blue")
plt.scatter(cf.ft.omega[:], hloc_constant[:], color="red")
plt.title("Hartree_Loc")
plt.xlabel("freq")
plt.ylabel("H_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()














print("**SigmaFLoc start")
floc = SigmaFLoc(crystal=cf.crystal, ft=cf.ft, occr=gloc.occ, vloc=vloc)
print("**SigmaFLoc finish\n")




# print("**SigmaFLoc start")
# floc = SigmaFLoc(crystal=cf.crystal, ft=cf.ft, occr=gloc.occ, vloc=vloc)
flat = SigmaFock(crystal=cf.crystal, occr=gint.occr, vbare=cf.vbare.r)
# print("**SigmaFLoc finish\n")

# print(flat.r.shape)
# print(floc.r.shape)

(norb, norb, ns, nr) = flat.r.shape
flat_average = np.zeros((norb, norb, ns, nf), dtype=np.complex128, order="F")
for iff in range(nf):
    # for ir in range(nr):
    #     flat_average[...,iff] += 1/nr * flat.r[...,ir]
    flat_average[..., iff] = flat.r[..., 0]


floc_constant = np.zeros(nf)
floc_constant[:] = floc.r[0,0,0,0]

plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], flat_average[0, 0, 0, :], color="blue")
plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
plt.title("Fock_Loc")
plt.xlabel("freq")
plt.ylabel("F_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()





imp={'temperature'            : 300, # temperature (in K)
     '1':
     {
      'impurity_matrix': [ # equivalent orbital index matrix. starting from 1.
         [1]
         ],       
     'thermalization_time': 3,
     'measurement_time': 20,
     'green_cutoff':  40,  
     'coulomb': 'full',
     }}



equiv = cf.crystal.read_imp_equi_mat(imp)

# equiv = np.array([[1,2,0],
#                   [2,1,0],
#                   [0,0,1]])













print("**Eimp start")
eimp = EImp(crystal=cf.crystal,niham=cf.niham,mu=cf.green.mu,hamh=cf.sigmah,hamf=cf.sigmaf,hloc=hloc,floc=floc)
print("**Eimp finish\n")

print("Eimp ")
for i in range(len(cf.crystal.probspace)):
    print(eimp.r[:,:,0,i])

print('*** Eimp B2F start ***')
eimp_F = eimp.imp_B2F(imp,eimp.r[...,0,0])
print('*** Eimp B2F finish ***')

print('*** Eimp F2B start ***')
eimp_B = eimp.imp_F2B(imp,eimp_F)
print('*** Eimp F2B finish ***')

print('*** Eimp_final_input start ***')
eimp.imp_final_input()
print('*** Eimp_final_input finish ***')



print("**Hybridisation start")
delta = Hybridisation(crystal=cf.crystal,ft=cf.ft,gloc=gloc,eimp=eimp,sigmahimp=hloc.r,sigmafimp=floc.r,sigmacimp=sigma_loc_gwc.rf)
print("**Hybridisation finish\n")


print('*** Hybridisation B2F start ***')
(norbc,norbc,ns,nft,nprob)=delta.rf.shape
# print(delta.rf.shape)
rf_temp = np.zeros((norbc,norbc,nft,nprob),dtype=np.complex128,order='F')
for iprob in range(nprob):
    for ifreq in range(nft):
        rf_temp[...,ifreq,iprob] = delta.rf[...,0,ifreq,iprob]
delta_F = delta.imp_B2F(imp,rf_temp[...,0])
print('*** Hybridisation B2F finish ***')

print('*** Hybridisation F2B start ***')
delta_B = delta.imp_F2B(imp,delta_F)
print('*** Hybridisation F2B finish ***')

# print(delta_B)


print('*** write_Hybridisation_json start ***')
hyb_dict = delta.write_dict_LocDyn(equiv,delta.rf[...,0])
delta.write_hyb_json(1,1,hyb_dict)
print('*** write_Hybridisation_json finish ***')


print('*** write_ctqmc_params start ***')
delta.write_ctqmc_params(1,1,eimp,equiv)
print('*** write_ctqmc_params finish ***')

delta.run_ctqmc()


exit()

plot = plt.figure(1)
plt.scatter(cf.ft.omega[1:], delta.rf[0, 0, 0, 1:, 0].real, color="blue")
plt.title("Hybridisation -- real part")
plt.xlabel("freq")
plt.ylabel("Delta")
# plt.ylim(-5, 5)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], delta.rf[0, 0, 0, :, 0].imag, color="blue")
plt.title("Hybridisation -- imag part")
plt.xlabel("freq")
plt.ylabel("Delta")
# plt.ylim(-5, 5)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()


