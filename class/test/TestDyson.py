import numpy as np
import sys
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE


norb = 4
ns = 2
nk = 5
nomega = 10

tempmat1 = np.zeros([norb,norb],dtype=complex,order='F')
tempmat2 = np.zeros([norb,norb],dtype=complex,order='F')
tempmat3 = np.zeros([8,8],dtype=complex,order='F')
tempmat4 = np.zeros([8,8],dtype=complex,order='F')
tempmat5 = np.zeros([8,8],dtype=complex,order='F')
tempmat6 = np.zeros([8,8],dtype=complex,order='F')
glatt0 = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
glattref = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
siglatt = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
glocstc = np.zeros([norb,norb,ns],dtype=complex,order='F')
glatstc = np.zeros([norb,norb,ns,nk],dtype=complex,order='F')
glocdyn = np.zeros([norb,norb,ns,nomega],dtype=complex,order='F')
glatdyn = np.zeros([norb,norb,ns,nk,nomega],dtype=complex,order='F')
wlatt0 = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
wlattref = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
platt = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
wlocstc = np.zeros([norb,norb,ns,ns],dtype=complex,order='F')
wlatstc = np.zeros([norb,norb,ns,ns,nk],dtype=complex,order='F')
wlocdyn = np.zeros([norb,norb,ns,ns,nomega],dtype=complex,order='F')
wlatdyn = np.zeros([norb,norb,ns,ns,nk,nomega],dtype=complex,order='F')
divisionarray = np.array([4,2],dtype=int,order='F')
ind = np.zeros([1,1],dtype=int,order='F')
nn1 = np.zeros([1,1],dtype=int,order='F')
nn2 = np.zeros([1,1],dtype=int,order='F')

omega = np.zeros([nomega],dtype=complex,order='F')
nu = np.zeros([nomega],dtype=complex,order='F')

for iomega in range(nomega):
    omega[iomega] = 1j*(2*iomega+1)

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    if iorb==jorb:
                        glatt0[iorb,jorb,js,ik,iomega] = 1.0/(omega[iomega]-1.0 +(ik+1)+(js+1)*0.1 + ((iorb+1)+(jorb+1))*2.0)

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    siglatt[iorb,jorb,js,ik,iomega] = 5*(ik+1)+(iorb+1)+0.1*(iomega+1)+(jorb+1)*2+(jorb+1)


            tempmat1 = DiagE.common.dcmplx_matinv(glatt0[:,:,js,ik,iomega],4)
            tempmat2 = tempmat1-siglatt[:,:,js,ik,iomega]
            glattref[:,:,js,ik,iomega] = DiagE.common.dcmplx_matinv(tempmat2,4)


glocstc = DiagE.dyson.flocstc(glatt0[:,:,:,0,0],siglatt[:,:,:,0,0])
glatstc = DiagE.dyson.flatstc(glatt0[:,:,:,:,0],siglatt[:,:,:,:,0])
glocdyn = DiagE.dyson.flocdyn(glatt0[:,:,:,0,:],siglatt[:,:,:,0,:])
glatdyn = DiagE.dyson.flatdyn(glatt0,siglatt)

print("Dyson FLocStc")

for js in range(ns):
    for iorb in range(norb):
        for jorb in range(norb):
            err = glattref[iorb,jorb,js,0,0]-glocstc[iorb,jorb,js]
            if abs(err) >= 1.0e-6:
                print(iorb,jorb,js,abs(err),glattref[iorb,jorb,js,0,0],glocstc[iorb,jorb,js])


print("Dyson FLatstc")

for ik in range(nk):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                err = glattref[iorb,jorb,js,ik,0] - glatstc[iorb,jorb,js,ik]
                if abs(err) >= 1.0e-6:
                    print(iorb,jorb,js,ik,abs(err),glattref[iorb,jorb,js,ik,0],glatstc[iorb,jorb,js,ik])

print("Dyson FLocDyn")

for iomega in range(nomega):
    for js in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                err = glattref[iorb,jorb,js,0,iomega]-glocdyn[iorb,jorb,js,iomega]
                if abs(err) >=1.0e-6:
                    print(iorb,jorb,js,iomega,abs(err),glattref[iorb,jorb,js,0,iomega],glocdyn[iorb,jorb,js,iomega])

print("Dyson FLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = glattref[iorb,jorb,js,ik,iomega]-glatdyn[iorb,jorb,js,ik,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,ik,iomega,abs(err),glattref[iorb,jorb,js,ik,iomega],glatdyn[iorb,jorb,js,ik,iomega])


for inu in range(nomega):
    nu[inu] = 1j*(2*inu)

ind1 = np.zeros([1],dtype=int,order='F') 
ind2 = np.zeros([1],dtype=int,order='F')

for inu in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        if iorb==jorb:
                            wlatt0[iorb,jorb,js,ks,ik,inu] = 1.0/(nu[inu]+1.0+(ik+1)+((js+1)-(ks+1))*0.1 +((iorb+1)+(jorb+1))*2.0)

for inu in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        platt[iorb,jorb,js,ks,ik,inu] = 5*(ik+1)+(iorb+1)+0.1*inu+(jorb+1)*2+(jorb+1)+((js+1)-(ks+1))*0.1

        tempmat3 = np.zeros([8,8],dtype=complex,order='F')
        tempmat4 = np.zeros([8,8],dtype=complex,order='F')
        for iorb in range(norb):
            for js in range(ns):
                nn1 = [iorb,js]
                nn1 = np.array(nn1,order='F')
                DiagE.common.indexing(8,divisionarray,1,ind1,nn1)
                for jorb in range(norb):
                    for ks in range(ns):
                        nn2 = [jorb,ks]
                        nn2 = np.array(nn2,order='F')
                        DiagE.common.indexing(8,divisionarray,1,ind2,nn2)
                        tempmat3[ind1,ind2] = wlatt0[iorb,jorb,js,ks,ik,inu]
                        tempmat4[ind1,ind2] = platt[iorb,jorb,js,ks,ik,inu]


        tempmat5 = DiagE.common.dcmplx_matinv(tempmat3,8)
        tempmat6 = tempmat5-tempmat4
        tempmat3 = DiagE.common.dcmplx_matinv(tempmat6,8)
       
        
        for iorb in range(norb):
            for js in range(ns):
                nn1 = [iorb,js]
                nn1 = np.array(nn1,order='F')
                DiagE.common.indexing(8,divisionarray,1,ind1,nn1)
                for jorb in range(norb):
                    for ks in range(ns):
                        nn2 = [jorb,ks]
                        nn2 = np.array(nn2,order='F')
                        DiagE.common.indexing(8,divisionarray,1,ind2,nn2)
                        wlattref[iorb,jorb,js,ks,ik,inu] = tempmat3[ind1,ind2]



wlocstc = DiagE.dyson.blocstc(wlatt0[:,:,:,:,0,0],platt[:,:,:,:,0,0])
wlatstc = DiagE.dyson.blatstc(wlatt0[:,:,:,:,:,0],platt[:,:,:,:,:,0])
wlocdyn = DiagE.dyson.blocdyn(wlatt0[:,:,:,:,0,:],platt[:,:,:,:,0,:])
wlatdyn = DiagE.dyson.blatdyn(wlatt0,platt)

print("Dyson BLocStc")

for js in range(ns):
    for ks in range(ns):
        for iorb in range(norb):
            for jorb in range(norb):
                err = wlattref[iorb,jorb,js,ks,0,0]-wlocstc[iorb,jorb,js,ks]
                if abs(err) >= 1.0e-6:
                    print(iorb,jorb,js,ks,abs(err),wlattref[iorb,jorb,js,ks,0,0],wlocstc[iorb,jorb,js,ks])

print("Dyson BLatStc")

for ik in range(nk):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = wlattref[iorb,jorb,js,ks,ik,0] - wlatstc[iorb,jorb,js,ks,ik]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,ks,ik,abs(err),wlattref[iorb,jorb,js,ks,ik,0],wlatstc[iorb,jorb,js,ks,ik])

print("Dyson BLocDyn")

for iomega in range(nomega):
    for js in range(ns):
        for ks in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    err = wlattref[iorb,jorb,js,ks,0,iomega]-wlocdyn[iorb,jorb,js,ks,iomega]
                    if abs(err) >= 1.0e-6:
                        print(iorb,jorb,js,ks,iomega,abs(err),wlattref[iorb,jorb,js,ks,0,iomega],wlocdyn[iorb,jorb,js,ks,iomega])

print("Dyson BLatDyn")

for iomega in range(nomega):
    for ik in range(nk):
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        err = wlattref[iorb,jorb,js,ks,ik,iomega]-wlatdyn[iorb,jorb,js,ks,ik,iomega]
                        if abs(err) >= 1.0e-6:
                            print(iorb,jorb,js,ks,ik,iomega,abs(err),wlattref[iorb,jorb,js,ks,ik,iomega],wlocdyn[iorb,jorb,js,ks,ik,iomega])
