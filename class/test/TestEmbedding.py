import numpy as np
import sys
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE

norb = 3
norb2 = 4
ns = 2
nomega = 10
nk = 5

gloc = np.zeros([norb,norb,ns,nomega],dtype=complex,order='F')
gproj = np.zeros([norb2,norb,ns,nk],dtype=complex,order='F')
glattref = np.zeros([norb2,norb2,ns,nk,nomega],dtype=complex,order='F')
glocstc = np.zeros([norb2,norb2,ns],dtype=complex,order='F')
glatsct = np.zeros([norb2,norb2,ns,nk],dtype=complex,order='F')
glocdyn = np.zeros([norb2,norb2,ns,nomega],dtype=complex,order='F')
tempmat1 = np.zeros([norb2,norb],dtype=complex,order='F')
wloc = np.zeros([norb,norb,ns,ns,nomega],dtype=complex,order='F')
wlattref = np.zeros([norb2,norb2,ns,ns,nk,nomega],dtype=complex,order='F')
wproj = np.zeros([norb2,norb,ns,nk],dtype=complex,order='F')
wlocstc = np.zeros([norb2,norb2,ns,ns],dtype=complex,order='F')
wlatstc = np.zeros([norb2,norb2,ns,ns,nk],dtype=complex,order='F')
wlocdyn = np.zeros([norb2,norb2,ns,ns,nomega],dtype=complex,order='F')
wlatdyn = np.zeros([norb2,norb2,ns,ns,nk,nomega],dtype=complex,order='F')



for iomega in range(nomega):
    for is1 in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                gloc[iorb,jorb,is1,iomega] = (iorb+1)  + 0.1*iomega+(jorb+1)*2 + 1j*((is1+1)*0.1)

for ik in range(nk):
    for is1 in range(ns):
        for iorb in range(norb2):
            for jorb in range(norb):
                gproj[iorb,jorb,is1,ik] = 0.1*(is1+1)+(ik+1)*0.5+(iorb-jorb)*2.0


for iomega in range(nomega):
    for ik in range(nk):
        for is1 in range(ns):
            tempmat1 = np.matmul(gproj[:,:,is1,ik],gloc[:,:,is1,iomega])
            glattref[:,:,is1,ik,iomega] = np.matmul(tempmat1, gproj[:,:,is1,ik].T.conj())
#for iorb in range(4):
#    for jorb in range(4):
#        print(glattref[iorb,jorb,0,0,0])               
gproj1 = np.array(gproj[:,:,:,0],order='F')


glocstc = DiagE.embedding.flocstc(gloc[:,:,:,0],gproj1)
glatstc = DiagE.embedding.flatstc(gloc[:,:,:,0],gproj)
glocdyn = DiagE.embedding.flocdyn(gloc,gproj1)
glatdyn = DiagE.embedding.flatdyn(gloc,gproj)

print("Embedding FLocStc")

for is1 in range(ns):
    for iorb in range(norb2):
        for jorb in range(norb2):
            err = glattref[iorb,jorb,is1,0,0]-glocstc[iorb,jorb,is1]
            if abs(err) >= 1.0e-6:
                print(iorb,jorb,is1,abs(err),glattref[iorb,jorb,is1,1,0],glocstc[iorb,jorb,is1])
print("Embedding FLocDyn")

for iomega in range(nomega):
    for is1 in range(ns):
        for iorb in range(norb2):
            for jorb in range(norb2):
                err = glattref[iorb,jorb,is1,0,iomega]-glocdyn[iorb,jorb,is1,iomega]
                if abs(err) >= 1.0e-6:
                   print(iorb, jorb, is1, iomega, abs(err), glattref[iorb,jorb,is1,0,iomega],glocdyn[iorb,jorb,is1,iomega])

print("Embedding FLatStc")

for ik in range(nk):
    for is1 in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                err = glattref[iorb,jorb,is1,ik,0]-glatstc[iorb,jorb,is1,ik]
                if abs(err) >= 1.0e-6:
                    print(iorb, jorb, is1, ik, abs(err), glattref[iorb,jorb,is1,ik,0],glatstc[iorb,jorb,is1,kt])

print("Embedding FLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for is1 in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = glattref[iorb,jorb,is1,ik,iomega]-glatdyn[iorb,jorb,is1,ik,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,is1,ik,iomega,glattref[iorb,jorb,is1,ik,iomega],glatdyn[iorb,jorb,is1,ik,iomega])

for iomega in range(nomega):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    wloc[iorb,jorb,js,ks,iomega] = (iorb+1)+0.1*iomega + (jorb+1)*2 + 1j*((js+1)-(ks+1))*0.1

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                wproj[iorb, jorb, js, ik] = 0.1*(js+1)+(ik+1)*0.5+((iorb+1)-(jorb+1))*2

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                tempmat1 = np.matmul(wproj[:,:,js,ik],wloc[:,:,js,ks,iomega])
                wlattref[:,:,js,ks,ik,iomega] = np.matmul(tempmat1,wproj[:,:,ks,ik].T.conj())

wlocstc = DiagE.embedding.blocstc(wloc[:,:,:,:,0],wproj[:,:,:,0])
wlatstc = DiagE.embedding.blatstc(wloc[:,:,:,:,0],wproj)
wlocdyn = DiagE.embedding.blocdyn(wloc,wproj[:,:,:,0])
wlatdyn = DiagE.embedding.blatdyn(wloc,wproj)


print("Embedding BLocStc")

for js in range(ns):
    for ks in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                err = wlocstc[iorb,jorb,js,ks] - wlattref[iorb,jorb,js,ks,0,0]
                if abs(err) >= 1.0e-6:
                    print(iorb,jorb,js,ks,abs(err),wlocstc[iorb,jorb,js,ks],wlattref[iorb,jorb,js,ks,0,0])

print("Embedding BLatStc")

for ik in range(nk):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = wlatstc[iorb, jorb, js, ks, ik] - wlattref[iorb,jorb,js,ks,ik,0]
                    if abs(err) >=1.0e-6:
                        print(iorb, jorb,js,ks,ik,abs(err),wlatstc[iorb,jorb,js,ks,ik],wlattref[iorb,jorb,js,ks,ik,0])


print("Embedding BLocDyn")

for iomega in range(nomega):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = wlocdyn[iorb,jorb,js,ks,iomega]-wlattref[iorb,jorb,js,ks,0,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,ks,iomega,abs(err),wlocdyn[iorb,jorb,js,ks,iomega],wlattref[iorb,jorb,js,ks,0,iomega])


print("Embedding BLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        err = wlatdyn[iorb,jorb,js,ks,ik,iomega]-wlattref[iorb,jorb,js,ks,ik,iomega]
                        if abs(err) >= 1.0e-6:
                            print(iorb,jorb,js,ks,ik,iomega,abs(err),wlatdyn[iorb,jorb,js,ks,ik,iomega],wlattref[iorb,jorb,js,ks,ik,iomega])






