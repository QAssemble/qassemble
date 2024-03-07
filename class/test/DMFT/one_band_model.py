import numpy as np
import matplotlib.pyplot as plt
import sys
from Newclass import CorrelationFunction, Crystal, NIHamiltonian
import time, datetime

lat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10]]
pos = {
    "CorF": "C",
    "pos": [
        [0, 0, 0],
    ],
}
# Hopping, single site, square grid.
t1 = 1.0
hopplist = [[t1, 0, 0, [1, 0, 0]], [t1, 0, 0, [0, 1, 0]]]
ns = 1
soc = False
rkgrid = [101, 101, 1]
orboption = [
    [0, 1],
]
# One electron in the unit cell
N = 1.0
impdict = {"1": [[[0, 0]]]}
print("Initialization start")
func = CorrelationFunction(lat, pos, ns, soc, rkgrid, orboption, N, impdict=impdict)
print("Initialization finish")
onsitelist = [0]
cry = Crystal(lat, pos, ns, soc, rkgrid, orboption, N, impdict=impdict)
ni_ham = NIHamiltonian(crystal=cry, hoppinglist=hopplist, onsitelist=onsitelist)
U = 10
slater_param = [U, 0, 0]
impurity_option = {
    1: {"KorS": "S", "value": slater_param, "site": 0, "orbitals": [0]},
}


max_iter = 100
mix = 1
T = 500
grid_size = 2000
print("DMFT start")
t0 = time.perf_counter()
func.DMFT(max_iter, ni_ham, impurity_option, N, mix, T, grid_size, [[1]])
t0 = time.perf_counter() - t0
print("DMFT finish")
print(f"DMFT loop time = {t0}")
