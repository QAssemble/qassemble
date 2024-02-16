import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/moseongjun/Desktop/Code/DiagE/class/')
from Newclass import *
import time, datetime
lat = [[1,0,0],[0,1,0],[0,0,1]]
pos = {"CorF" : "F","pos" : [[0.5,0.5,0.5]]}
ns = 1
soc = False
rkgrid = [10,10,1]
orboption = [[0,1]]
N = 0.5
func = CorrelationFunction(lat,pos,ns,soc,rkgrid,orboption,N)
U = 2
Vij = 0.4 
option = {1: {"KorS" : "S", "value" : [U,0,0],"site" : 0, "orbitals" : [0]}}
V = [[Vij,0,0,[1,0,0]],[Vij,0,0,[-1,0,0]],[Vij,0,0,[0,1,0]],[Vij,0,0,[0,-1,0]]]
t = -0.05
hoppinglist = [[t,0,0,[1,0,0]],[t,0,0,[0,1,0]]]
onsitelist = [0]
iter = 100
mix = 0.05
T = 1/(8.6173303*10**-5*100)
size = 1000
print("GW Start")
start = time.time()
func.GWApproximation(iter,mix,T,size,hoppinglist,onsitelist,option,V)
end = time.time()
print("GW finish")
delta = datetime.timedelta(seconds=(end-start))
print(f"GW loop time : {delta}")
green = func.green
gf = np.zeros((1,1,1,1000),dtype=complex,order="F")
for ir in range(10*10):
    gf += green.gkf[...,ir,:]
gf/=100
print(gf[0,0,0,0].imag)
plt.plot(green.ft.omega[1::],gf[0,0,0,1::].imag)
plt.show()
