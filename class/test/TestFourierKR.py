import numpy as np
import sys
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE

norb = 3
ns = 2
nk = 125
nomega = 10


fr = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
fk = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
fk2 = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
bk = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
br = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
bk2 = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
irk = 0
ind = np.zeros([1,1,1],dtype=int,order='F')
divisionarray = np.array([5,5,5],dtype=int,order='F')

for iomega in range(nomega):
    for kk in range(5):
        for jk in range(5):
            for ik in range(5):
                ind = [ik+1,jk+1,kk+1]
                ind = np.array(ind,order='F')
                DiagE.common.indexing(nk,divisionarray,1,irk,ind)

                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            fk[iorb,jorb,js,irk,iomega] = ((iorb+1)-(jorb+1))/2.0 +(js+1)*0.1 + (ik+jk+kk)/2.0 + (iomega+1)*0.001

fr = DiagE.fourier.flatdyn_k2r(divisionarray,fk)
fk2 = DiagE.fourier.flatdyn_r2k(divisionarray,fr)

print("Fourier Fermion KR")

for iomega in range(nomega):
    for irk in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = fk[iorb,jorb,js,irk,iomega]-fk2[iorb,jorb,js,irk,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,irk,iomega,fk[iorb,jorb,js,irk,iomega],fk2[iorb,jorb,js,irk,iomega])

ikr = 0

for iomega in range(nomega):
    for kk in range(5):
        for jk in range(5):
            for ik in range(5):
                ind = [ik+1,jk+1,kk+1]
                ind = np.array(ind,order='F')
                DiagE.common.indexing(nk,divisionarray,1,ikr,ind)

                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                bk[iorb,jorb,js,ks,ikr,iomega] = ((iorb+1)-(jorb+1))/2.0 + ((js+1)-(ks+1))*0.1+(ik+jk+kk)/2.0 + iomega*0.001

br = DiagE.fourier.blatdyn_k2r(divisionarray,bk)
bk2 = DiagE.fourier.blatdyn_r2k(divisionarray,br)

print("Fourier Boson KR")

for iomega in range(nomega):
    for ikr in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        err = bk[iorb,jorb,js,ks,ikr,iomega]-bk2[iorb,jorb,js,ks,ikr,iomega]
                        if abs(err) >= 1.0e-6:
                            print(iorb,jorb,js,ks,ikr,iomega,abs(err),bk[iorb,jorb,js,ks,ikr,iomega],bk2[iorb,jorb,js,ks,ikr,iomega])
