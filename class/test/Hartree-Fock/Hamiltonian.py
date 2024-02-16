import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/moseongjun/Desktop/Code/DiagE/class/')
from Newclass import *
import time, datetime

lat = [[2.4253866,0.0,0.0],[0,10,0],[0.0,0.0,10]]
pos = {"CorF" : "C", "pos" : [[0,0,0.5],[ 1.15498,-0.69896,0.5]]}
ns = 1
soc = False
rkgrid = [101,1,1]
orboption = [[0,1],[1,1]]
N = 1
# cry = Crystal(lat,pos,ns,soc,rkgrid,orboption,N)
print("Initialization start")
func = CorrelationFunction(lat,pos,ns,soc,rkgrid,orboption,N)
print("Initialization finish")
t1 = -2.568
t2 = -2.232
hopplist = [[t1,0,1,[0,0,0]],[t2,1,0,[1,0,0]]]
# t = -2.7
# t = -2.
# hopplist = [[t,0,1,[0,0,0]],[t,1,0,[1,0,0]],[t,1,0,[0,1,0]]]
onsitelist = [0,0]
print("Tight-Binding start")
hamtb = func.TighBinding(hoppinglist=hopplist,onsitelist=onsitelist)
print("Tight-Binding finish")
cry = Crystal(lat,pos,ns,soc,rkgrid,orboption,N)
temp = FLatStc(cry)
print("Plot band")
# energytb = temp.Diagonalize(hamtb)
# temp.Band(energytb,'./png/tb.png')
# plt.clf()
print("Plot band")
U = 8
option = {1: {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbitals" : [0]}, 2: {"KorS" : "S", "value" : [U,0,0], "site" : 0, "orbitals" : [1]}}
k = 2.0

#Ohno parameterization
V = []
Rmax = 101
Rmin = lambda R1, R2: R1 if R1<R2 else R2
for ii in range(2):
    for jj in range(2):
       for cnt in range(0,Rmax):
              print(ii,jj)
              if ii != jj :
                     print("other atom basis")
                     if cnt == 0:
                        rvec = [cnt,0,0]
                        # rvec = [cnt%Rmax,0,0]
                        delta = func.cry.basisc[ii,:] - (func.cry.basisc[jj,:] + np.array(rvec)*2.4253866)
                        R1 = np.linalg.norm(delta)
                        print(R1,rvec)
                        R2 = np.linalg.norm(delta+np.array([Rmax,0,0])*2.4253866)
                        
                        R = Rmin(R1,R2)
                        # R = R1
                        print(R)
                        # V.append([U/(k*np.sqrt(1+0.6117*R**2))*1/2,ii,jj,rvec])
                        V.append([U/(k*np.sqrt(1+0.6117*R**2))*1,ii,jj,rvec])
                     else:
                        rvec = [cnt,0,0]
                        # rvec = [cnt%Rmax,0,0]
                        delta = func.cry.basisc[ii,:] - (func.cry.basisc[jj,:] + np.array(rvec)*2.4253866)
                        R1 = np.linalg.norm(delta)
                        print(R1,rvec)
                        R2 = np.linalg.norm(delta+np.array([Rmax,0,0])*2.4253866)
                        
                        R = Rmin(R1,R2)
                        # R = R1
                        print(R)
                        # V.append([U/(k*np.sqrt(1+0.6117*R**2))*1/2,ii,jj,rvec])
                        V.append([U/(k*np.sqrt(1+0.6117*R**2))*1,ii,jj,rvec])


              if ii == jj:
                     if cnt == 0:
                        continue
                     print("same atom basis")
                     rvec = [cnt,0,0]
                     # rvec = [cnt%Rmax,0,0]
                     delta = func.cry.basisc[ii,:] - (func.cry.basisc[jj,:] + np.array(rvec)*2.4253866)
                     
                     R1 = np.linalg.norm(delta)
                     print(R1,rvec)
                     R2 = np.linalg.norm(delta+np.array([Rmax,0,0])*2.4253866)
                     
                     R = Rmin(R1,R2)
                     # R = R1
                     print(R)
                     # V.append([U/(k*np.sqrt(1+0.6117*R**2))*1/2,ii,jj,rvec])
                     V.append([U/(k*np.sqrt(1+0.6117*R**2))*1,ii,jj,rvec])
              else:
                   continue
              
e1 = 0
e2 = 0
for v in V:
    if v[1]==0:
        e1 += v[0]
    elif v[1]==1:
        e2 += v[0]
e1 += U
e2 += U
onsitelist = [-e1,-e2]
print(onsitelist)

iter = 100
mix = 1
T = 300
size = 1000
print("HF start")
start = time.time()
hamhf, sigmah, sigmaf = func.HartreeFockH(iter,mix,T,size,hopplist,onsitelist,option,V)
end = time.time()
print("HF finish")
delta = datetime.timedelta(seconds=(end-start))
print(f"HF loop time = {delta}")
energytest, eigvec = temp.Diagonalize(hamhf,True)
temp.Band(energytest)
print(abs(energytest[0,0].max()-energytest[1,1].min()))

cry.Kpath([[0,0,0],[0.5,0,0]],100)

# Interpolation
def linear(x,x0,y0,x1,y1):
  L0 = np.linalg.norm(x1-x)/np.linalg.norm(x1-x0)
  L1 = np.linalg.norm(x-x0)/np.linalg.norm(x1-x0)
  y = y0*L0 + y1*L1 
  return y

def find_closest_row(arr, input_arr):
    # Calculate the Euclidean distances between input_arr and each row in arr
    distances = np.linalg.norm(arr - input_arr, axis=1)
    idx1 = np.argmin(distances)
    arr2 = np.delete(arr,idx1,axis=0)
    dumdist = np.linalg.norm(arr2-input_arr,axis=1)
    idxdum = np.argmin(dumdist)
    dist2 = np.linalg.norm(arr-arr2[idxdum],axis=1)
    idx2 = np.argmin(dist2)
    arr3 = np.delete(arr2,idxdum,axis=0)
    dumdist2 = np.linalg.norm(arr3-input_arr,axis=1)
    idxdum2 = np.argmin(dumdist2)
    dist3 = np.linalg.norm(arr-arr3[idxdum2],axis=1)
    idx3 = np.argmin(dist3)

    return idx1,arr[idx1],idx2,arr[idx2],idx3,arr[idx3]

# Symmetry Check
plot = np.zeros((2,Rmax),dtype=float)
for ik in range(Rmax):
    for iorb in range(2):
        plot[iorb,ik] = energytest[iorb,iorb,0,ik]
plot = plot.T
rev_plot = plot[::-1,...]
# plt.plot(plot[0:int(Rmax/2)])
# plt.plot(rev_plot[0:int(Rmax/2)])

plot_band = np.zeros((100,2),dtype=float)
for ik in range(100):
    ii,xii,jj,xjj,kk,xkk = find_closest_row(cry.kpoint,cry.kpath[ik])
    for iorb in range(2):
        plot_band[ik,iorb] = linear(cry.kpath[ik],xii,plot[ii,iorb],xjj,plot[jj,iorb])
        # plot_band[ik,iorb] = quadratic(cry.kpath[ik],xii,plot[ii,iorb],xjj,plot[jj,iorb],xkk,plot[kk,iorb])
bandppp = np.loadtxt('band_ppp.dat',dtype=float)
plt.plot(bandppp[:,0],bandppp[:,1::],'k-')
print(abs(bandppp[:,1].max()-bandppp[:,2].min()))
plt.plot(bandppp[:,0],plot_band,'r--')
print(abs(plot_band[:,0].max()-plot_band[:,1].min()))
plt.savefig('validationH.png')
np.savetxt('band_diageH.dat',plot_band)
