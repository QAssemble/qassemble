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
NElec = 3 # 1
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


i = 1
for size in [9,2]:
    i = i*size

print(i)

# ind1, [iorb,js] = cf.crystal.indexing(9*2,2,[9,2],0,0,[0,0])

# print(ind1, iorb,js)

exit()














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














print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Start computing Local Double-CounGreenLocting GW")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

from qacore.BLocDyn import BLocDyn, PolLGW, WLoc,WLoc_temp
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

gint = cf.green
cf.crystal.Projector(impdict={"1": [[[0, 0],[0, 1],[0, 2]]]})

# vbare.vloc.GetUijklComCTQMC('1')
# print(vbare.vloc.u_ctqmc)
# exit()
# impdict = {
#     "1" : [
#         [
#             [0, 5],
#             [0, 6],
#             [0, 7],
#             [0, 8],
#             [0, 9]
#         ]
#     ]
# }
# cf.crystal.Projector(impdict)
# cf.crystal.Projector(impdict={"1": [[[0, 0],[0, 1]]]})
# cf.crystal.Projector(impdict = {"1" : [[[0, 1]]], "2" : [[[1, 2], [1,3]]], "3" : [[[2, 2]]]})
# exit()
# print(cf.crystal.fprojector.shape)
# print(cf.crystal.bprojector.shape)
import pprint

pprint.pprint(cf.crystal.bimpdict)
print(cf.crystal.bimpdict['1'][0])
print(len(cf.crystal.bimpdict['1'][0]))
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

vloc2 = np.zeros_like(cf.vbare.k)
(norb, norb, ns, ns, nk) = cf.vbare.k.shape
nspace = cf.crystal.fprojector.shape[3]
vloc = np.zeros((norb, norb, ns, ns, nspace), dtype=np.complex128, order="F")
for ik in range(nk):
    vloc2[..., ik] = cf.vbare.vloc.vloc
# vloc2[...,0] = cf.vbare.vloc.vloc
vloc[..., 0] = vloc2[..., 0]
# vloc[..., 1] = vloc2[0, 0, 0, 0, 0]
# print("vloc.shape --", vloc.shape)
# print(vloc[:, :, 0, 0, 0])
# print("vbare.shape --", cf.vbare.k.shape)
# print(vloc2[:, :, 0, 0, 0])
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
# start = time.time()
polloc_dc = PolLGW(crystal=cf.crystal, ft=cf.ft, green=gloc)
# end = time.time()
# tiem_delta = end-start
# print(datetime.timedelta(seconds=tiem_delta))
pollat = PolLat(crystal=cf.crystal, ft=cf.ft, green=g_average2)
# polloc_average = PolLoc(crystal=cf.crystal,ft=cf.ft,green=g_average)
# print('polLoc.rt.shape --',polloc.rt.shape)
print("**PolLoc finish\n")
# exit()
# (norb,norb, ns, ns, nk, nf) = cf.pol.kf.shape
# pol_average = np.zeros((norb, norb, ns, ns, nf),dtype=np.complex128, order='F')
# for i in range(nk):
#     pol_average += 1/nk*cf.pol.kf[...,i,:]

# for iff in range(nf):
#     for js in range(ns):
#         for ks in range(ns):
#             for jorb in range(1):
#                 for iorb in range(1):
#                     # err = pol_average[iorb, jorb, js, ks, iff] - polloc.rf[iorb, jorb, js, ks, iff, 0]
#                     err = (
#                         pollat.rf[iorb, jorb, js, ks, 0, iff]
#                         - polloc_dc.rf[iorb, jorb, js, ks, iff, 0]
#                     )
#                     if abs(err) > 1.0e-6:
#                         print(
#                             iorb,
#                             jorb,
#                             js,
#                             ks,
#                             iff,
#                             abs(err),
#                             pollat.rf[iorb, jorb, js, ks, 0, iff],
#                             polloc_dc.rf[iorb, jorb, js, ks, iff, 0],
#                         )

plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], pollat.rf[0, 0, 0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.nu[:],polloc_average.rf[0,0,0,0,:],color='blue')
plt.scatter(cf.ft.nu[:], polloc_dc.rf[0, 0, 0, 0, :, 0], color="red")
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


# print(pollat.kf[..., 0:5, 0])

# print(polloc_dc.rf[..., 0, 0])
kf = np.zeros_like(pollat.kf, dtype=np.complex128, order='F')
for ik in range(nk):
    kf[..., ik, :] = pollat.rf[..., 0, :]


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
wloc = WLoc_temp(crystal=cf.crystal, ft=cf.ft, pol=polloc_dc.rf, vLoc=vloc)
# wloc = WLoc(crystal=cf.crystal, ft=cf.ft, wlat=cf.w)
# wlat = WLat(crystal=cf.crystal, ft=cf.ft, pol=pollat.kf, vbare=cf.vbare)
wlat = WLat_k(crystal=cf.crystal, ft=cf.ft, pol=kf, vbare=vloc2)
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
plt.scatter(cf.ft.nu[:], wloc.crf[0, 0, 0, 0, :, 0], color="red")
plt.title("Wc_Loc")
plt.xlabel("freq")
plt.ylabel("Wc_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

# plot = plt.figure(1)
# plt.scatter(cf.ft.nu[:], wlat.rf[0, 0, 0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.nu[:],wlat_average[0,0,0,0,:],color='blue')
# plt.scatter(cf.ft.nu[:], wloc.rf[0, 0, 0, 0, :, 0], color="red")
# plt.title("W_Loc")
# plt.xlabel("freq")
# plt.ylabel("W")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()

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


from qacore.BLocDyn import BLocDyn, PolLGW, WLoc
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC,Sigma_imp
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

from qacore.FLocStc import EImp
from qacore.FLocDyn import Hybridisation,Weiss_Green
from qacore.BLocDyn import UImp

print(vloc.shape)

print("**Uimp start")
uimp = UImp(crystal=cf.crystal,ft=cf.ft,wloc=wloc.rf,ploc=polloc_dc.rf,vloc=vloc)
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

plot = plt.figure(1)
plt.scatter(cf.ft.nu[1:], uimp.ubar_rf[0, 0, 0, 0, 1:, 0].real, color="blue")
plt.title("ubar -- real part")
plt.xlabel("freq")
plt.ylabel("Delta")
# plt.ylim(-5, 5)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()

# print("**WLoc start")
# wloc = WLoc(crystal=cf.crystal, ft=cf.ft, pol=polloc.rf, vLoc=vloc, vDyn=uimp.utilde_rf)
# print("**WLoc finish\n")
# plot = plt.figure(1)
# plt.scatter(cf.ft.nu[:], wloc.rf[0, 0, 0, 0, :, 0], color="blue")
# # plt.scatter(cf.ft.nu[:],wlat_average[0,0,0,0,:],color='blue')
# plt.scatter(cf.ft.nu[:], wloc_temp.rf[0, 0, 0, 0, :, 0], color="red")
# plt.title("W_Loc")
# plt.xlabel("freq")
# plt.ylabel("W")
# plt.legend()
# plt.grid(which="both", linestyle="--", linewidth=0.3)
# plt.show()



print("**SigmaLGWC start")
sigma_loc_dc = SigmaLGWC(crystal=cf.crystal, ft=cf.ft, green=gloc.gt, wloc=wloc.crt)
print("**SigmaLGWC finish\n")

plot = plt.figure(1)
# plt.scatter(cf.ft.omega[:], sigma_lat_gwc.rf[0, 0, 0, 0, :], color="blue")
# # plt.scatter(cf.ft.omega[:],sigmagwc_lat_average[0,0,0,:],color='blue')
plt.scatter(cf.ft.omega[:], sigma_loc_dc.rf[0, 0, 0, :, 0], color="red")
plt.title("Sigma")
plt.xlabel("freq")
plt.ylabel("Sigma")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()



# exit()





print("**SigmaHLoc start")
hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=uimp.utilde_rf)
# hloc = SigmaHLoc(crystal=cf.crystal, occ=gloc.occ, vloc=vloc)
print("**SigmaHLoc finish\n")

# print(hloc.r)
# exit()


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
         [1,0,0],
         [0,1,0],
         [0,0,1]
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


print('hloc')
print(hloc.r[...,0])
print('floc')
print(floc.r[...,0])




# print("***Weiss Green's function start")
# weiss_green = Weiss_Green(crystal=cf.crystal,ft=cf.ft,niham=cf.niham,mu=cf.green.mu,hamh=cf.sigmah,hamf=cf.sigmaf,hloc=hloc,floc=floc,gloc=gloc,sigmahimp=hloc.r,sigmafimp=floc.r,sigmacimp=sigma_loc_dc.rf)
# print("***Weiss Green's function finish")



print("\n**Eimp start")
eimp = EImp(crystal=cf.crystal,niham=cf.niham,mu=cf.green.mu,hamh=cf.sigmah,hamf=cf.sigmaf,hloc=hloc,floc=floc)
print("**Eimp finish")

# print("Eimp ")
# for i in range(len(cf.crystal.probspace)):
#     print(weiss_green.Eimp_r[:,:,0,i])

print('*** Eimp B2F start ***')
# eimp_F = weiss_green.imp_B2F(imp,weiss_green.Eimp_r[...,0,0])
eimp_F = eimp.imp_B2F(imp,eimp.r[...,0,0])
print('*** Eimp B2F finish ***')

print('*** Eimp F2B start ***')
eimp_B = eimp.imp_F2B(imp,eimp_F)
print('*** Eimp F2B finish ***')

print('*** Eimp_final_input start ***')
eimp.imp_final_input()
print('*** Eimp_final_input finish ***')



print("\n**Hybridisation start")
delta = Hybridisation(crystal=cf.crystal,ft=cf.ft,gloc=gloc,eimp=eimp,sigmahimp=hloc.r,sigmafimp=floc.r,sigmacimp=sigma_loc_dc.rf)
print("**Hybridisation finish")

print('*** Hybridisation B2F start ***')
(norbc,norbc,ns,nft,nprob)=delta.rf.shape
# print(delta.rf.shape)
rf_temp = np.zeros((norbc,norbc,nft,nprob),dtype=np.complex128,order='F')
for iprob in range(nprob):
    for ifreq in range(nft):
        rf_temp[...,ifreq,iprob] = delta.rf[...,0,ifreq,iprob]
delta_F = delta.imp_B2F_freq(imp,rf_temp[...,0])
print('*** Hybridisation B2F finish ***')

print('*** Hybridisation F2B start ***')
delta_B = delta.imp_F2B_freq(imp,delta_F)
print('*** Hybridisation F2B finish ***')

# print(delta_B)


print('\n*** write_Hybridisation_json start ***')
hyb_dict = delta.write_dict_LocDyn(equiv,delta.rf[...,0])
# sigma_imp=Sigma_imp(crystal=cf.crystal,ft=cf.ft)
delta.write_hyb_json(1,1,hyb_dict)
print('*** write_Hybridisation_json finish ***')


print("\n*** write_Dyn_json start ***")
print("*** compute F0 start ***")
norb,_,ns,_,nft,_ = uimp.utilde_rf.shape
norbc = len(cf.crystal.find)

utilde_rf_4 = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nft),dtype=np.complex64,order='F')

for iis in range(ns):
    for jjs in range(ns):
        for ift in range(nft):
            utilde_rf_4[...,iis,jjs,ift] = cf.crystal.Double2Quad(uimp.utilde_rf[...,iis,jjs,ift,0])

F0_val = np.zeros(nft,dtype=np.float64, order='F')
for ift in range(nft):
    F0_val[ift] = 1.0/ns**2/norbc**2*np.einsum('ijjimn->',utilde_rf_4[...,ift]).real
print("*** compute F0 finish ***")
F0_dict = {}
F0_dict["F0"] = F0_val.tolist()

delta.write_dyn_json(1,1,F0_dict)

print("*** write_Dyn_json finish ***")


print('\n*** write_ctqmc_params start ***')
delta.write_ctqmc_params(1,1,eimp,equiv,vloc)
print('*** write_ctqmc_params finish ***')


# exit()

print('\n*** run and measure CTQMC start ***')
# delta.run_ctqmc()
# delta.measure_ctqmc()
print('*** run and measure CTQMC finish ***')


print('\n*** impurity postprocessing start ***')
green_edmft_freq, sigmac_edmft_freq, sigmahf_edmft, Chi_edmft_4, histo = delta.impurity_postprocessing(1,1,equiv)
print('*** impurity postprocessing finish ***')

# exit()





plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], green_edmft_freq[0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
plt.title("Green's function")
# plt.xlabel("freq")
# plt.ylabel("F_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()


plot = plt.figure(1)
plt.scatter(cf.ft.omega[:], sigmac_edmft_freq[0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
plt.title("Sigma_C")
# plt.xlabel("freq")
# plt.ylabel("F_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()



plot = plt.figure(1)
plt.scatter(cf.ft.nu[:], Chi_edmft_4[0, 0, 0, 0, 0, 0, :], color="blue")
# plt.scatter(cf.ft.omega[:], floc_constant[:], color="red")
plt.title("Chi")
# plt.xlabel("freq")
# plt.ylabel("F_Loc")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.3)
plt.show()



























print("\n*** Pi_edmft start ***")

norbc,_,_,_,ns,_,nft = Chi_edmft_4.shape
Chi_edmft_temp = np.zeros((norbc*norbc,norbc*norbc,2,2,nft),dtype=np.complex64,order='F')

for iis in range(ns):
    for jjs in range(ns):
        for ift in range(nft):
            Chi_edmft_temp[:,:,iis,jjs,ift] = cf.crystal.Quad2Double(Chi_edmft_4[:,:,:,:,iis,jjs,ift])

Chi_edmft_2 = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nft),dtype=np.complex64,order='F')

for ift in range(nft):
    Chi_edmft_2[...,0,0,ift] = 1.0/2.0*(Chi_edmft_temp[...,0,0,ift]+Chi_edmft_temp[...,0,1,ift]+Chi_edmft_temp[...,1,0,ift]+Chi_edmft_temp[...,1,1,ift])

# print('Pi_edmft_2 done')



def compute_Pi(X, u):
    """
    Compute Pi(ω) = (I + X(ω) u(ω))^(-1) X(ω) for each frequency.

    Parameters
    ----------
    X : (n, n, nfreq) ndarray
    u : (n, n, nfreq) ndarray

    Returns
    -------
    Pi : (n, n, nfreq) ndarray
    """
    n, _, ns, _, nfreq = X.shape
    Pi = np.zeros_like(X, dtype=complex)
    I = np.eye(n, dtype=X.dtype)

    for iis in range(ns):
        for jjs in range(ns):
            for iw in range(nfreq):
                A = I + X[:, :, iis, jjs, iw] @ u[:, :, iis, jjs, iw, 0]
                Pi[:, :, iis, jjs, iw] = np.linalg.solve(A, X[:, :, iis, jjs, iw])

    return Pi


# print(green_edmft_freq.shape)
# print(uimp.utilde_rf.shape)
# print(Chi_edmft_2.shape)

# Pi_edmft = compute_Pi(Chi_edmft_2,uimp.utilde_rf)


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




# exit()

# print(cf.crystal.ft.omega)
# print(cf.crystal.ft.nu)
# print(cf.crystal.ft.tau)

# print(green_edmft_freq.shape)
# print(sigmac_edmft_freq.shape)
# print(sigmahf_edmft.shape)

norbc = cf.crystal.fprojector.shape[1]
ns = cf.crystal.ns
nft = len(cf.ft.omega)
ntau = len(cf.ft.tau)

# print(norbc,ns,nft,ntau)


green_edmft_tau = np.zeros((norbc,norbc,ns,ntau), dtype=np.complex128, order='F')
green_edmft_tau = gloc.F2T(green_edmft_freq,1,1) ### ?

# print('toto')

rho = (-1) * green_edmft_tau[:, :, :, -1].copy()         # (i,j,s)
# Enforce Hermiticity per spin
rho = 0.5 * (rho + rho.swapaxes(0, 1).conj())

# print('toto')

def hartree_Sigma_diag_density_density(rho, V):
    """
    Diagonal Hartree Σ_H for density–density v:
      v[i,j,k,l,s,sp] = δ_{ij} δ_{kl} V[i,l,s,sp]

    Inputs
      G: (norbs, norbs, nspin, ntau)
      V: (norbs, norbs, nspin, nspin)   # V[i,l,s,sp] couples n_i^s to n_l^sp
      t0: int
      sign: +1 if rho =  G(0), -1 if rho = -G(0^-)

    Returns
      Sigma_H: (norbs, norbs, nspin)  # diagonal in orbital space
    """
    norb, _, nspin = rho.shape
    # rho = _rho_from_G_equal_time(G, t0, sign=sign)         # (i,j,s)
    # occupations per spin: n[l,sp]
    n_occ = np.real(np.stack([np.diag(rho[:, :, sp]) for sp in range(nspin)], axis=1))

    # Σ_diag[i,s] = sum_{l,sp} V[i,l,s,sp] * n_occ[l,sp]
    Sigma_diag = np.einsum('ilsp,lp->is', V, n_occ, optimize=True)

    Sigma_H = np.zeros((norb, norb, nspin), dtype=Sigma_diag.dtype)
    idx = np.arange(norb)
    Sigma_H[idx, idx, :] = Sigma_diag
    return Sigma_H

def hartree_Sigma_diag_general(rho, v):
    """
    Diagonal Hartree Σ_H from general spin-resolved 4-index v.

    Inputs
      G: (norbs, norbs, nspin, ntau)             # G[i,j,s,t]
      v: (norbs, norbs, norbs, norbs, nspin, nspin)  # v[i,j,k,l,s,sp]
      t0: int  # time index for t -> 0
      sign: +1 if rho =  G(0), -1 if rho = -G(0^-)

    Returns
      Sigma_H: (norbs, norbs, nspin)  # diagonal in orbital space
    """
    norb, _, nspin = rho.shape
    # rho = _rho_from_G_equal_time(G, t0, sign=sign)   # (i,j,s)
    I = np.eye(norb)

    # Σ_diag[i,s] = sum_{j,k,l,sp} v[i,j,k,l,s,sp] * rho[l,k,sp] * δ_{ij}
    # Sigma_diag = np.einsum('ijklsp,lkp,ij->is', v, rho, I, optimize=True)
    Sigma_diag = np.einsum('ijklmn,jkn,il->im', v, rho, I, optimize=True)

    # Pack into diagonal matrices for each spin
    Sigma_H = np.zeros((norb, norb, nspin), dtype=Sigma_diag.dtype)
    idx = np.arange(norb)
    Sigma_H[idx, idx, :] = Sigma_diag
    return Sigma_H

# sigmah_edmft = hartree_Sigma_diag_density_density(rho, uimp.utilde_rf[...,0,0])


print("\n*** sigmah_edmft start ***")

v_temp = np.zeros((norbc,norbc,norbc,norbc,ns,ns), dtype=np.complex128, order='F')
for iis in range(ns):
    for jjs in range(ns):
        v_temp[...,iis,jjs] = cf.crystal.Double2Quad(uimp.utilde_rf[...,iis,jjs,0,0])

sigmah_edmft = hartree_Sigma_diag_general(rho, v_temp)

# print("sigmahf_edmft")
# print(sigmahf_edmft)
print("*** sigmah_edmft finish ***")
# print(sigmah_edmft)


print("*** sigmaf_edmft start ***")
sigmaf_edmft = np.zeros((norbc,norbc,ns), dtype=np.complex128, order='F')
for i in range(ns):
    sigmaf_edmft[...,i] = sigmahf_edmft[...,i] - sigmah_edmft[...,i]

print("*** sigmaf_edmft finish ***")
# print(sigmaf_edmft)

# print('toto')

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


