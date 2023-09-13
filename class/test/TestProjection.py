import numpy as np
import DiagE

norb = 4
norbc = 3
ns = 2
nk = 5
nomega = 10

tempmat1 = np.zeros([norb, norbc],dtype=complex,order='F')
glatt = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
gproj = np.zeros([norb,norbc,ns,nk],dtype=complex,order='F')
glocref = np.zeros([norbc,norbc,ns,nk,nomega],dtype=complex,order='F')
glocstc = np.zeros([norbc,norbc,ns],dtype=complex,order='F')
glocdyn = np.zeros([norbc,norbc,ns,nomega],dtype=complex,order='F')
glatstc = np.zeros([norbc,norbc,ns,nk],dtype=complex,order='F')
glatdyn = np.zeros([norbc,norbc,ns,nk,nomega],dtype=complex,order='F')
wlatt = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
wproj = np.zeros([norb,norbc,ns,nk],dtype=complex,order='F')
wlocref = np.zeros([norbc,norbc,ns,ns,nk,nomega],dtype=complex,order='F')
wlocstc = np.zeros([norbc,norbc,ns,ns],dtype=complex,order='F')
wlocdyn = np.zeros([norbc,norbc,ns,ns,nomega],dtype=complex,order='F')
wlatstc = np.zeros([norbc,norbc,ns,ns,nk],dtype=complex,order='F')
wlatdyn = np.zeros([norbc,norbc,ns,ns,nk,nomega],dtype=complex,order='F')

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    glatt[iorb,jorb,js,ik,iomega] = 10*(ik+1)+(iorb+1)+0.1*(iomega+1)+(jorb+1)*2+1j*(js+1)*0.1


for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norbc):
                gproj[iorb,jorb,js,ik] = 0.1*(js+1) +(ik+1)*0.5 +((iorb+1)-(jorb+1))*2.0


for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            tempmat1 = np.matmul(glatt[:,:,js,ik,iomega],gproj[:,:,js,ik])
            glocref[:,:,js,ik,iomega] += np.matmul(gproj[:,:,js,ik].T.conj(),tempmat1)

glocstc = DiagE.projection.flocstc(glatt[:,:,:,0,0],gproj[:,:,:,0])
glatstc = DiagE.projection.flatstc(glatt[:,:,:,:,0],gproj)
glocdyn = DiagE.projection.flocdyn(glatt[:,:,:,0,:],gproj[:,:,:,0])
glatdyn = DiagE.projection.flatdyn(glatt,gproj)


print("Projectoin FLocStc")

for js in range(ns):
    for iorb in range(norbc):
        for jorb in range(norbc):
            err = glocstc[iorb,jorb,js] - glocref[iorb,jorb,js,0,0]
            if abs(err) >= 1.0e-6:
                print(iorb,jorb,js,abs(err),glocstc[iorb,jorb,js], glocref[iorb,jorb,js,0,0])


print("Projection FLatStc")

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norbc):
            for jorb in range(norbc):
                err = glatstc[iorb,jorb,js,ik]-glocref[iorb,jorb,js,ik,0]
                if abs(err) >=1.0e-6:
                    print(iorb,jorb,js,ik,abs(err),glatstc[iorb,jorb,js,ik],glocref[iorb,jorb,js,ik,0])

print("Projection FLocDyn")

for iomega in range(nomega):
    for js in range(ns):
        for iorb in range(norbc):
            for jorb in range(norbc):
                err = glocdyn[iorb,jorb,js,iomega]-glocref[iorb,jorb,js,0,iomega]
                if abs(err) >=1.0e-6:
                    print(iorb,jorb,js,iomega,abs(err),glocdyn[iorb,jorb,js,iomega],glocref[iorb,jorb,js,0,iomega])

print("Projection FLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norbc):
                for jorb in range(norbc):
                    err = glatdyn[iorb,jorb,js,ik,iomega]-glocref[iorb,jorb,js,ik,iomega]
                    if abs(err)>=1.0e-6:
                        print(iorb,jorb,js,ik,iomega,abs(err),glatdyn[iorb,jorb,js,ik,iomega],glocref[iorb,jorb,js,ik,iomega])


for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        wlatt[iorb,jorb,js,ks,ik,iomega] = 10*(ik+1)+(iorb+1)*0.1*(iomega+1)+jorb*2+1j*((js+1)-(ks+1))*0.1

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norbc):
                wproj[iorb,jorb,js,ik] = 0.1*(js+1)+(ik+1)*0.5+((iorb+1)-(jorb+1))*2.0

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                tempmat1 = np.matmul(wlatt[:,:,js,ks,ik,iomega],wproj[:,:,ks,ik])
                wlocref[:,:,js,ks,ik,iomega] += np.matmul(wproj[:,:,js,ik].T.conj(),tempmat1)


wlocstc = DiagE.projection.blocstc(wlatt[:,:,:,:,0,0],wproj[:,:,:,0])
wlatstc = DiagE.projection.blatstc(wlatt[:,:,:,:,:,0],wproj)
wlocdyn = DiagE.projection.blocdyn(wlatt[:,:,:,:,0,:],wproj[:,:,:,0])
wlatdyn = DiagE.projection.blatdyn(wlatt,wproj)

print("Projection BLocStc")

for js in range(ns):
    for ks in range(ns):
        for iorb in range(norbc):
            for jorb in range(norbc):
                err = wlocstc[iorb,jorb,js,ks]-wlocref[iorb,jorb,js,ks,0,0]
                if abs(err) >= 1.0e-6:
                    print(iorb,jorb,js,ks,abs(err),wlocstc[iorb,jorb,js,ks],wlocref[iorb,jorb,js,ks,0,0])

print("Projection BLatStc")

for ik in range(nk):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norbc):
                for jorb in range(norbc):
                    err=wlatstc[iorb,jorb,js,ks,ik]-wlocref[iorb,jorb,js,ks,ik,0]
                    if abs(err) >=1.0e-6:
                        print(iorb,jorb,js,ks,ik,abs(err),wlatstc[iorb,jorb,js,ks,ik],wlocref[iorb,jorb,js,ks,ik,0])

print("Projection BLocDyn")

for iomega in range(nomega):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norbc):
                for jorb in range(norbc):
                    err = wlocdyn[iorb,jorb,js,ks,iomega]-wlocref[iorb,jorb,js,ks,0,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,ks,iomega,abs(err),wlocdyn[iorb,jorb,js,ks,iomega],wlocref[iorb,jorb,js,ks,0,iomega])

print("Projection BLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norbc):
                    for jorb in range(norbc):
                        err = wlatdyn[iorb,jorb,js,ks,ik,iomega]-wlocref[iorb,jorb,js,ks,ik,iomega]
                        if abs(err) >= 1.0e-6:
                            print(iorb,jorb,js,ks,ik,iomega,abs(err),wlatdyn[iorb,jorb,js,ks,ik,iomega]-wlocref[iorb,jorb,js,ks,ik,iomega])
