import numpy as np
import DiagE
from DiagE import bare as bare

norb = 4
ns = 2
nk = 5
nomega = 100

ai = 1j
beta = 1.0/(8.617333262145e-5*300.0)
pi = np.pi

fhlatt = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')
fflatt = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
omega = np.zeros([nomega],dtype=complex,order='F')
tempmat1 = np.zeros([norb,norb],dtype=complex,order='F')
tempmat2 = np.zeros([norb,norb],dtype=complex,order='F')

for i in range(nomega):
    omega[i] = pi/beta*(2*i+1)

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                if iorb==jorb:
                   fhlatt[iorb,jorb,js,ik] = 1.0 +(ik+1)+(js+1)*0.2
                else:
                   fhlatt[iorb,jorb,js,ik] = 0.1+(ik+1)+(js+1)+((iorb+1)+(jorb+1))*0.1
for ik in range(nk):
    for js in range(ns):
        tempmat1 = fhlatt[:,:,js,ik]
        E = np.linalg.eigvalsh(tempmat1) 
      
        gfreq = np.zeros([nomega,norb],dtype=complex,order='F')
        for iorb in range(norb):
            gfreq[:,iorb] = bare.ffreq(omega,E[iorb])

for ik in range(nk):
    for js in range(ns):
        tempmat1 = fhlatt[:,:,js,ik]
        tempmat1 = np.array(tempmat1,order='F')
        E = np.linalg.eigvalsh(tempmat1.T)
        gfreq2 = np.zeros([nomega,norb],dtype=complex,order='F')

        for iorb in range(norb):
            for iomega in range(nomega):
                gfreq2[iomega, iorb] = 1.0/(ai*omega[iomega]-E[iorb])

for iomega in range(nomega):
    for iorb in range(norb):
        err = gfreq[iomega,iorb]-gfreq2[iomega,iorb]
        if abs(err) >= 1.0e-6:
           print(iorb,iomega,abs(err),gfreq[iomega,iorb],gfreq2[iomega,iorb])
