import numpy as np
import sys
from scipy.linalg import blas as FB 
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE

itheta = 0.0
norb = 4
ns = 2
nk = 5
nomega = 1000
ntau = nomega

ai = 0+1j
beta = 1.0/(8.617333262145e-5*300.0)
pi = np.pi

fhlatt = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')
fflatt = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
fflatt2 = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')

omega = np.zeros([nomega],dtype=complex,order='F')
#gfreq = np.zeros([nomega,norb],dtype=complex,order='F')
tempmat1 = np.zeros([norb,norb],dtype=complex,order='F')
tempmat2 = np.zeros([norb,norb],dtype=complex,order='F')

err = 0.0+0.0j


for i in range(nomega):
    omega[i] = pi/beta*(2*i+1)
            
for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                if iorb==jorb:
                    fhlatt[iorb,jorb,js,ik] = 1.0+(ik+1)+(js+1)*0.1
                else:
                    fhlatt[iorb,jorb,js,ik] = 0.1 +(ik+1)+(js+1)*0.1 + ((iorb+1)+(jorb+1))*0.1
                                              
for ik in range(nk):
    for js in range(ns):
        tempmat1 = fhlatt[:,:,js,ik].T
        E,tempmat1 = np.linalg.eig(tempmat1) 
        gfreq = np.zeros([nomega,norb],dtype=complex,order='F') 

        for iorb in range(norb):
            for iomega in range(nomega):
                gfreq[iomega, iorb] = 1.0/(ai*omega[iomega]-E[iorb])

        for iomega in range(nomega):
            for iorb in range(norb):
                for jorb in range(norb):
                    tempmat2[iorb,jorb] = tempmat1[iorb,jorb]*gfreq[iomega,jorb]

            fflatt[:,:,js,ik,iomega] = np.matmul(tempmat2,tempmat1.T)
                 
#print(np.shape(E))
#print(fflatt)
fhlatt2 = np.array(fhlatt.T,order='F')
fflatt2 = DiagE.bare.flatfreq(fhlatt,omega)
#print(fflatt2)
print("Bare fermion frequency test")
for ik in range(nk):
    for iomega in range(nomega):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = fflatt[iorb,jorb,js,ik,iomega]-fflatt2[iorb,jorb,js,ik,iomega]
                    if (abs(err) >= 1.0e-8):
                        print(iorb,jorb,js,ik,iomega,abs(err),fflatt[iorb,jorb,js,ik,iomega],fflatt2[iorb,jorb,js,ik,iomega])
#########################Fermion Tau####################################
tau = np.zeros([ntau],dtype=complex,order='F')
ftlatt = np.zeros([norb,norb,ns,nk,ntau],dtype=complex,order='F')
ftlatt2 = np.zeros([norb,norb,ns,nk,ntau],dtype=complex,order='F')
fmoment = np.zeros([norb,norb,ns,nk,3],dtype=complex,order='F')
fhigh = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')
err = 0.0
itheta = 0.0

for itau in range(ntau):
    itheta = DiagE.common.ttind(itau,ntau)
    tau[itau] = beta/2.0*(np.cos(pi*(itheta+0.5)/ntau)+1)
energy = 0.1
gtau = DiagE.bare.ftau(tau,energy,omega)

for itau in range(ntau):
    print(np.real(tau[itau]),np.real(gtau[itau]))
