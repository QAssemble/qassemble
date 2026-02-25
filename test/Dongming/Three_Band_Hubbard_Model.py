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

from qacore.FPathStc import FPathStc

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
# KGrid = [10, 10, 10]
NElec = 2 # 3 # 1 ## should be 1 for validation
beta = 100
# T = 2000
cutoff =  30 #50 #30#40#30 #100 #30
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
cf = CorrelationFunction(cry=cry, ft=ft)

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

fpathstc = FPathStc(crystal=cf.crystal, obj=cf.niham)


# fpathstc.Dos(matr=cf.niham.r,plotoption=True)


kpath = [[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]]
nkpath = 101
fpathstc.crystal.Kpath(kpath, nkpath)



U = 1 #5 #1
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




# itermax = 1
# mix = 0.1



# # start = time.time()

# # cf.GWApproximation(
# #     itermax=itermax,
# #     mix=mix,
# #     hoppinglist=hopping,
# #     loccoulomb=locoption,
# #     nonloccoulomb=nonlocoption,
# # )

# # end = time.time()
# # tiem_delta = end-start
# # print("\n==============================================================")
# # print("GWApproximation code      - ",round(tiem_delta,5),'seconds')
# # print("==============================================================")


# start = time.time()

# cf.GWApproximation_new(
#     itermax=itermax,
#     mix=mix,
#     hoppinglist=hopping,
#     loccoulomb=locoption,
#     nonloccoulomb=nonlocoption,
# )

# end = time.time()
# tiem_delta = end-start
# print("\n==============================================================")
# print("GWApproximation new code      - ",round(tiem_delta,5),'seconds')
# print("==============================================================")

# fpathstc.Band(hmat = cf.niham.r+cf.sigmah.r+cf.sigmaf.r, plotoption=True)


# exit()










from qacore.BLocDyn import BLocDyn, PolLGW, WLoc,WLoc_temp
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc
import pprint


from qacore.BLocDyn import BLocDyn, PolLGW, WLoc
from qacore.FLocDyn import FLocDyn, GreenLoc, SigmaLGWC
from qacore.FLocStc import FLocStc, SigmaFLoc, SigmaHLoc

from qacore.FLocStc import EImp
from qacore.FLocDyn import Hybridisation,FWeiss
from qacore.BLocDyn import BWeiss
from qacore.BLocStc import VLoc

from qacore.CTQMC import CTQMC








cf.crystal.Projector(impdict={"1": [[[0, 0],[0, 1],[0, 2]]]})

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


itermax = 2
mix = 0.1

cf.GW_EDMFT(
    itermax=itermax,
    imp=imp,equiv=equiv,
    mix=mix,
    hoppinglist=hopping,
    loccoulomb=locoption,
    nonloccoulomb=nonlocoption,
)









