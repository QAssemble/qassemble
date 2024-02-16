import numpy as np
import sys, os
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE
norb = 3
ns = 2
nk = 5
nomega = 300
ntau = nomega

tempmat = np.zeros([norb,norb],dtype=complex,order='F')
tempmat1 = np.zeros([norb,norb],dtype=complex,order='F')
tempmat2 = np.zeros([norb,norb],dtype=complex,order='F')
eig = np.zeros([norb],dtype=complex,order='F')
eigmat = np.zeros([norb,norb],dtype=complex,order='F')
hmat = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')
identity = np.identity(norb,dtype=complex)
identity = np.array(identity,order='F')
glatt0 = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
flatdyn_t = np.zeros([norb,norb,ns,nk,ntau],dtype=complex,order='F')
flatdyn_f = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
flatdyn_moment = np.zeros([norb,norb,ns,nk,3],dtype=complex,order='F')
flatdyn_high = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')

omega = np.zeros([nomega],dtype=float,order='F')
tau = np.zeros([ntau],dtype=float,order='F')

beta = 1.0/(8.617333262145e-5*300.0)
pi = np.pi

for iomega in range(nomega):
    omega[iomega] = pi/beta*(2*iomega+1)

for itau in range(ntau):
    itheta = DiagE.common.ttind(itau,ntau)
    tau[itau] = beta/2.0*(np.cos(pi*(itheta+0.5)/ntau)+1)

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                hmat[iorb,jorb,js,ik] = ((iorb+1)+(jorb+1))*0.5+(js+1)*0.3*(ik+1)
                if iorb==jorb:
                    hmat[iorb,jorb,js,ik] = hmat[iorb,jorb,js,ik]-6.0

for ik in range(nk):
    for js in range(ns):
        for iomega in range(nomega):
            tempmat1 = identity*omega[iomega]*1j-hmat[:,:,js,ik]
            tempmat2 = DiagE.common.dcmplx_matinv(tempmat1,norb)
            glatt0[:,:,js,ik,iomega] = tempmat2


flatdyn_moment, flatdyn_high = DiagE.fourier.flatdyn_m(omega,glatt0,1,1)
flatdyn_t = DiagE.fourier.flatdyn_f2t(omega,glatt0,flatdyn_moment,tau)

flatdyn_f = DiagE.fourier.flatdyn_t2f(tau,flatdyn_t,omega)

print("Fourier FLatDynFT")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = glatt0[iorb,jorb,js,ik,iomega]-flatdyn_f[iorb,jorb,js,ik,iomega]
                    if abs(err)>=1.0e-2:
                        print(iorb,jorb,js,ik,iomega,abs(err),glatt0[iorb,jorb,js,ik,iomega],flatdyn_f[iorb,jorb,js,ik,iomega])


