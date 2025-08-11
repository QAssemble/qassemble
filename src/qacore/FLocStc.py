import string as string
from typing import Any
import matplotlib as mat
import re as re
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
import json, os, shutil, sys
import itertools
import scipy.optimize
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.fftpack import fftn, ifftn
import scipy.linalg
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
import subprocess
import copy
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatStc import NIHamiltonian,SigmaHartree,SigmaFock, Hamiltonian
# from .FLocDyn import GreenLoc
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/qacore/modules')
import QAFort

class FLocStc(object):

    def __init__(self,crystal : Crystal):

        self.crystal = crystal

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            matinv[:,:,js] = np.linalg.inv(mat[:,:,js])

        return matinv
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]

        Fnew = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        Fnew = mix*Fb + (1.0-mix)*Fold

        return Fnew
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]


        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nspace),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
        
    def Loc2Imp(self,matloc : np.ndarray)-> np.ndarray:

        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[3]

        matimp = np.zeros((norb,norb,ns,nprob),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns),dtype=np.complex128)
            for ispace in val:
                tempmat += matloc[...,ispace]
            tempmat /=len(val)
            matimp[...,iprob] = tempmat

        return matimp
    
    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:

        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        
        for ind in range(nind):
            matdict[ind+1] = []
            pos = self.crystal.FindPositions(equiv,ind+1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    e += matin[ii,jj,js]
                e /= len(pos)
                matdict[ind+1].append(e)
        
        return matdict
    
    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv,ind+1)
                for ii,jj in pos:
                    matout[ii,jj,js] = matdict[str(ind+1)]

        return matout
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

       
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        
        matout = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        matout = QAFort.dyson.flocstc(mat1,mat2)

        return matout 

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nspace = self.crystal.fprojector.shape[3]
        
        matout = np.zeros((norb,norb,ns,nrk),dtype=np.complex128,order='F')
        
        for ispace in range(nspace):
            matout += QAFort.embedding.flocstc(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

        return matout





    def imp_B2F(self,imp,B):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index

        nprob = len(self.crystal.probspace)

        F = {}
        
        for ii in range(nprob):

            iimp = str(ii+1)

            F[iimp] = {}
            for i in range(len(self.crystal.imp_index[ii])):
                index_of_equivalance = str(i+1)
                F[iimp][index_of_equivalance] = 0

                for j in range(len(self.crystal.imp_index[ii][i])):
                    F[iimp][index_of_equivalance] = F[iimp][index_of_equivalance] + B[self.crystal.imp_index[ii][i][j][0],self.crystal.imp_index[ii][i][j][1]]

                F[iimp][index_of_equivalance] = F[iimp][index_of_equivalance]/len(self.crystal.imp_index[ii][i]) ### take the average
            
        return F
    

    def imp_F2B(self,imp,F):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index
        
        nprob = len(self.crystal.probspace)
        norbc = self.crystal.fprojector.shape[1]

        B = np.zeros((norbc,norbc,nprob), dtype=np.complex128, order='F')
        ii = 0 # index of impurity problems -- nprob
        for key,val in F.items():
            i=0 # index of equivalances
            for valkey,valval in val.items():
                for j in range(len(self.crystal.imp_index[ii][i])):
                    B[self.crystal.imp_index[ii][i][j][0],self.crystal.imp_index[ii][i][j][1],ii] = valval
                i += 1
            ii += 1
        
        return B
    

    











    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]

        if os.path.exists('flocstc'):
            pass
        else:
            os.mkdir("flocstc")
        os.chdir("flocstc")
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, Re(F), Im(F)\n")
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        f.write(f"{iorb} {jorb} {js} {matin[iorb,jorb,js].real} {matin[iorb,jorb,js].imag}\n")
        os.chdir("..")
        return None
    
    
class ImpurityLevel(FLocStc):

    def __init__(self, crystal: Crystal, niham : NIHamiltonian, mu : float):
        super().__init__(crystal)
        
        self.niham = niham
        self.mu = mu
        self.loc = None
        self.imp = None
        self.Cal()

    def Cal(self):
        
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspace = self.crystal.fprojector.shape[3]

        ham = self.niham.UpdateMu(self.niham.k,self.mu)

        eimp = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            eimp[...,ispace] = QAFort.projection.flatstc(ham,self.crystal.fprojector[...,ispace])

        self.loc = eimp
        self.imp = self.Loc2Imp(eimp)

        return None
        

class SigmaHLoc(FLocStc):

    def __init__(self, crystal: Crystal, occ, vloc : object, hdf5file : str = 'glob.h5', group :str = None):
        super().__init__(crystal)
        
    #     
        self.r = None
        # self.ft = ft
        # self.k = None
        self.vloc = vloc ## frequency dependent V, utilde_rf
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.occ = occ

        self.Cal()
        # self.MakeDyn()

    def Cal(self):
        # vbare = self.vbare.k
        occ = self.occ
        # vk = self.vbare.Double2Quad(self.vbare.k)
        norbc = self.crystal.fprojector.shape[1]  # occk.shape[0]
        ns = self.crystal.ns  # occk.shape[2]
        # nk = len(self.crystal.kpoint)  # occk.shape[3]
        norb = self.crystal.bprojector.shape[1]  # vbare.shape[0]  ### ???
        # norb = len(self.crystal.bind)  # vbare.shape[?0]
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        # nf = len(self.ft.omega)

        # onsite = self.R2K(self.onsiter)
        h = np.zeros((norbc, norbc, ns, nprob), dtype=np.complex128, order="F")

        if self.crystal.ns != 1:
            #     for ik in range(nk):
            #         tempmat[...,ik] = self.crystal.OrbSpin2Composite(vbare[...,ik])

            # for ik in range(nk):
            #     for ind1 in range(norb*ns):
            #         nn1 = [0]*2
            #         ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
            #         [iorbc1,iorbc2] = self.crystal.b2f[iorb]

            #         for ind2 in range(norb*ns):
            #             nn2 = [0]*2
            #             ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
            #             [iorbc3,iorbc4] = self.crystal.b2f[jorb]
            #             h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks]
            # for jk in range(nk):
            #     h[iorbc1,iorbc2,js,ik] += tempmat[ind1,ind2,0]*occ[iorbc4,iorbc3,ks,jk]/nk
            for iprob in range(nprob):
                # for iff in range(nf):
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = self.crystal.indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a, m1])
                    iorbc2 = self.crystal.FIndex([a, m2])
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind2, nn2
                        )
                        [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b, m3])
                        iorbc4 = self.crystal.FIndex([b, m4])
                        # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]
                        h[iorbc1, iorbc2, js, iprob] += (
                            self.vloc[iorb, jorb, js, ks, 0, iprob] * occ[iorbc4, iorbc3, ks] ### vloc take only iomega = 0
                    )

        else:
            if self.crystal.soc == True:
                C = 1
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             # gtemp = np.zeros((norbc,norbc,1),dtype=np.complex64)
                #             # for jk in range(nk):
                #             #     gtemp[iorbc4,iorbc3,0] += g0kt[iorbc4,iorbc3,0,0,-1]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C #1/nk*gtemp[iorbc4,iorbc3,0]*C
                for iprob in range(nprob):
                    # for iff in range(nf):
                    for ind1 in range(norb * ns):
                        nn1 = [0] * 2
                        ind1, [iorb, js] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind1, nn1
                        )
                        [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a, m1])
                        iorbc2 = self.crystal.FIndex([a, m2])
                        for ind2 in range(norb * ns):
                            nn2 = [0] * 2
                            ind2, [jorb, ks] = self.crystal.indexing(
                                norb * ns, 2, [norb, ns], 0, ind2, nn2
                            )
                            [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b, m3])
                            iorbc4 = self.crystal.FIndex([b, m4])
                            h[iorbc1, iorbc2, js, iprob] = (
                                self.vloc[iorb, jorb, js, ks, 0, iprob]
                                * occ[iorbc4, iorbc3, ks]
                                * C
                            )

            else:
                C = 2
                # for ik in range(nk):
                #     for iorb in range(norb):
                #         iorbc1,iorbc2 = self.crystal.b2f[iorb]
                #         for jorb in range(norb):
                #             iorbc3, iorbc4 = self.crystal.b2f[jorb]
                #             h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0]*C
                #             # for jk in range(nk):
                #             #     h[iorbc1,iorbc2,0,ik] += vbare[iorb,jorb,0,0,0]*occ[iorbc4,iorbc3,0,jk]/nk*C
                for iprob in range(nprob):
                    # for iff in range(nf):
                    for ind1 in range(norb * ns):
                        nn1 = [0] * 2
                        ind1, [iorb, js] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind1, nn1
                        )
                        [a, [m1, m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a, m1])
                        iorbc2 = self.crystal.FIndex([a, m2])
                        for ind2 in range(norb * ns):
                            nn2 = [0] * 2
                            ind2, [jorb, ks] = self.crystal.indexing(
                                norb * ns, 2, [norb, ns], 0, ind2, nn2
                            )
                            [b, [m3, m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b, m3])
                            iorbc4 = self.crystal.FIndex([b, m4])
                            # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                            h[iorbc1, iorbc2, js, iprob] += (
                                self.vloc[iorb, jorb, js, ks, 0, iprob]
                                * occ[iorbc4, iorbc3, ks]
                                * C
                            )

        self.r = h  # +onsite
        # self.r = self.K2R(h)

        return None
    
    def MakeDyn(self): ### ??

        norb = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.gloc.gf.shape[3]
        nspace = self.crystal.fprojector.shape[3]

        hdyn = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex128,order='F')
        
        for ift in range(nft):
            hdyn[...,ift,:] = self.hloc
        
        self.hdyn = hdyn

        return None

class SigmaHImp(FLocStc):

    def __init__(self, crystal: Crystal):
        super().__init__(crystal)
        self.Cal()

    def Cal(self):
        pass

# class SigmaFLoc(FLocStc):

#     def __init__(self, crystal: Crystal, ft: FTGrid, occ, vloc : object, hdf5file : str = 'glob.h5', group :str = None):
#         super().__init__(crystal)

#         self.gloc = gloc
#         self.vbare = vbare
#         self.floc = None
#         self.fimp = None
#         self.fdyn = None
    
#         self.Cal()
#         self.MakeDyn()

#     def Cal(self):
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         norb = self.crystal.bprojector.shape[1]
#         nspace = self.crystal.fprojector.shape[3]

#         U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')
#         floc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')
        

#         for ispace in range(nspace):
#             U[...,ispace] = QAFort.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])

#             for js in range(ns):
#                 for iorb in range(norb):
#                     iorbc1, iorbc4 = self.crystal.b2f[iorb]
#                     for jorb in range(norb):
#                         iorbc3, iorbc2 = self.crystal.b2f[jorb]
#                         floc[iorbc1,iorbc2,js,ispace] += self.gloc.gf[iorbc4,iorbc3,js,-1,ispace]*U[iorb,jorb,js,js,ispace]

#         self.floc = floc
#         self.fimp = self.Loc2Imp(floc)
        
#         return None
    


class SigmaFLoc(FLocStc):

    def __init__(
        self,
        crystal: Crystal,
        ft: FTGrid,
        occr=None,
        vloc: np.ndarray = None,
        hdf5file: str = "glob.h5",
        group: str = None,
    ):  # green -> occ
        super().__init__(crystal)
        self.r = None
        # self.k = None
        self.ft = ft

        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        # self.green = green

        self.occr = occr
        self.vloc = vloc

        self.Cal()
        # self.MakeDyn()

    def Cal(self):

        # g0rt = self.green.glatrt
        occr = self.occr
        # vr = self.vbare.Double2Quad(self.vbare.r)

        # norbc = len(self.crystal.find)
        # ns = occr.shape[2]
        # nr = occr.shape[3]
        # norb = len(self.crystal.bind)
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.crystal.bprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        nf = len(self.ft.omega)

        fr = np.zeros((norbc, norbc, ns, nprob), dtype=np.complex128, order="F")

        # for ir in range(nr):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             [iorbc1,iorbc4] = self.crystal.b2f[iorb]
        #             for jorb in range(norb):
        #                 [iorbc2,iorbc3] = self.crystal.b2f[jorb]
        #                 fr[iorbc1,iorbc3,js,ir] = -occr[iorbc4,iorbc2,js,ir]*vr[iorb,jorb,js,js,ir]
        for iprob in range(nprob):
            # for iff in range(nf):
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = self.crystal.indexing(
                        norb * ns, 2, [norb, ns], 0, ind1, nn1
                    )
                    [a, [m1, m4]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a, m1])
                    iorbc4 = self.crystal.FIndex([a, m4])
                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = self.crystal.indexing(
                            norb * ns, 2, [norb, ns], 0, ind2, nn2
                        )
                        [b, [m3, m2]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b, m3])
                        iorbc2 = self.crystal.FIndex([b, m2])
                        if js == ks:
                            # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]
                            fr[iorbc1, iorbc2, js, iprob] += (
                                -occr[iorbc4, iorbc3, js, iprob]
                                * self.vloc[iorb, jorb, js, ks, iprob]
                            )

                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc3,iorbc4,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]

        # fk = self.R2K(fr)

        self.r = fr
        # self.k = fk
        # del fr, occr
        return None

    def Save(self, fn: str):

        # os.chdir('work')

        # filepath = 'flatstc.h5'
        # groupname = 'sigmaf'
        # with h5py.File(filepath,'a') as file:
        #     if self.CheckGroup(filepath,groupname):
        #         group = file[groupname]
        #     else:
        #         group=file.create_group(groupname)

        #     group.create_dataset(fn,dtype=complex,data=self.k)
        # os.chdir('..')
        with h5py.File(self.hdf5file, "a") as file:
            if self.CheckGroup(self.hdf5file, self.group):
                group = file[self.group]
                if self.subgroup in group:
                    sigmaf = group[self.subgroup]
                else:
                    sigmaf = group.create_group(self.subgroup)
            else:
                group = file.create_group(self.group)
                sigmaf = group.create_group(self.subgroup)
            sigmaf.create_dataset(fn, dtype=complex, data=self.k)

        return None



# class SigmaFImp(FLocStc):

#     def __init__(self, crystal: Crystal):
#         super().__init__(crystal)
#         self.Cal()

#     def Cal(self):
#         pass



class EImp(FLocStc):

    def __init__(self, crystal: Crystal, niham : NIHamiltonian = None, mu = None, hamh : SigmaHartree = None, hamf : SigmaFock = None, hloc : SigmaHLoc = None, floc : SigmaFLoc = None):
        super().__init__(crystal)


        # self.ham = ham
        self.niham = niham
        self.hamh = hamh
        self.hamf = hamf
        self.hloc = hloc
        self.floc = floc

        # print('hartree_loc')
        # print(self.hloc.r[:,:,0,0])
        # print('fock_loc')
        # print(self.floc.r[:,:,0,0])

        self.mu = mu

        self.hamhf = self.niham.k + self.hamh.k + self.hamf.k
        self.sigmahfloc = self.hloc.r + self.floc.r

        self.r = None
        self.A_final = None
        self.ctqmc_mu = None

        self.Cal()

    def Cal(self):

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.hamhf.shape[0]
        nk = self.hamhf.shape[3]
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        self.hamhf_projection = np.zeros_like(self.sigmahfloc)


        tempmat = np.zeros((norbc, norbc, ns, nspace), dtype=np.complex128, order='F')
        hamhf = np.copy(self.hamhf)
        
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    hamhf[iorb, iorb, js, ik] -= self.mu
                    # self.hamhf[iorb, iorb, js, ik] -= self.mu
        

        for ispace in range(nspace):
            tempmat[...,ispace] = QAFort.projection.flatstc(hamhf,self.crystal.fprojector[...,ispace])
        
        # print('hf_projection')
        # print(tempmat[:,:,0,0])
        # exit()
        # tempmat = self.niham.Projection(self.hamhf)
        self.hamhf_projection = self.Loc2Imp(tempmat)

        # print('hamhf_projection --'+str(self.hamhf_projection[0,0,0,0]))
        
        # I = np.identity(norbc)

        self.r = np.zeros((norbc,norbc,ns,nprob), dtype=np.complex128, order='F')
        for iis in range(ns):
            for iprob in range(nprob):
                self.r[...,iis,iprob] = self.hamhf_projection[...,iis,iprob] - self.sigmahfloc[...,iis,iprob] #- self.mu*I


    def imp_final_input(self): ## move to EImp

        nprob = len(self.crystal.probspace)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]

        if ns==1:
            mu = np.zeros(nprob, dtype=np.complex128, order='F')
            # for i in range(nprob):
            #     mu[i] = -B[0,0,i]  ### is the mu the same along the omega space?
            I = np.identity(len(self.r))
            A = np.zeros((norbc,norbc), dtype=np.complex128, order='F')
            A_final = np.zeros((norbc*2,norbc*2,nprob), dtype=np.complex128, order='F')

            for i in range(nprob):
                mu[i] = -self.r[0,0,0,i]
                A = self.r[...,0,i] + mu[i]*I
                A_final[...,i] = np.kron(A,np.eye(2,2))

            self.A_final = np.copy(A_final)
            self.ctqmc_mu = np.copy(mu)
        
        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
        
        return None


    # # def write_ctqmc_params(self,iter,key,E_imp : np.ndarray,equiv : np.ndarray):
    # def write_ctqmc_params(self,iter,key,equiv : np.ndarray):
        
    #     if self.crystal.soc is False:
    #         if self.crystal.ns ==1:
    #             params = {}
    #             params["hloc"] = {}

    #             # mu_ctqmc=-np.real(E_imp[0,0,0])
    #             # # print(mu_ctqmc,type(mu_ctqmc))
    #             # E_imp = E_imp[:,:,0]+mu_ctqmc*np.eye(E_imp.shape[0],E_imp.shape[0])
    #             # E_imp = np.array(np.real(E_imp),dtype=float)
    #             # tempmat = np.kron(E_imp,np.eye(2,2))

    #             params["hloc"]['one body'] = self.A_final[...,key-1].tolist()

    #             # self.boson.get_Uijkl_comctqmc(key)
    #             params["hloc"]["two body"]=0 # self.boson.U_ctqmc.tolist()
                
    #             params["partition"]={}
                
    #             params["partition"]["green basis"]= "matsubara"
    #             params["partition"]["green bulla"]= True
    #             params["partition"]["green matsubara cutoff"] = 50
    #             params["partition"]["occupation susceptibility bulla"]=True
    #             params["partition"]["occupation susceptibility direct"]=False
    #             params["partition"]["quantum number susceptibility"] = True
    #             params["partition"]["susceptibility cutoff"]=50
    #             params["partition"]["susceptibility tail"]=200
    #             params["partition"]["quantum numbers"]={}
    #             tempmat = np.ones(self.A_final[...,key-1].shape[0]*2)
    #             params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
    #             for ii in range(len(tempmat)):
    #                 if ii < self.A_final[...,key-1].shape[0]:
    #                     tempmat[ii]*= 0.5
    #                 elif ii >= self.A_final[...,key-1].shape[0]:
    #                     tempmat[ii]*=-0.5
    #             params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
    #             params["partition"]["probabilities"]={}
    #             params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
    #             params["partition"]["density matrix precise"] = True
    #             params["partition"]["print eigenstates"] = True
    #             params["partition"]["print density matrix"]= True
                

    #             params["beta"]=0 #self.ft.beta  ##### ft.beta ??
    #             params["complex"] = False
    #             params["mu"]=self.ctqmc_mu[key-1]
    #             params["hybridisation"]={}

    #             tempmat2 = np.kron(equiv,np.eye(2,2))  ###### equiv ??
    #             tempmat2 = tempmat2.tolist()
    #             for ii in range(len(tempmat2)):
    #                 for jj in range(len(tempmat2)):
    #                     if tempmat2[ii][jj]==0.0:
    #                         tempmat2[ii][jj] = ""
    #                     else:
    #                         tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

    #             params["hybridisation"]["matrix"]=tempmat2
    #             params["hybridisation"]["functions"]="hyb.json"
    #             params["thermalisation time"]=1 #imp['thermalization_time']
    #             params["quantum number susceptibility"]=True
    #             params["occupation susceptibility bulla"]=True        
    #             params["green bulla"]=True       
    #             params["density matrix precise"]=False #True 
    #             params["measurement time"]=3 #imp['measurement_time']
                
    #             with open(f'params.{iter}.{key}.json','w') as outfile:
    #                 json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
    #             with open('params.json','w') as outfile:
    #                 json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
    #             # print("params.json written", file=self.m_ini.control['h_log'])
    #         elif self.crystal.ns == 2:
    #             print("Nspin is not 1")
    #             sys.exit()
    #     elif self.crystal.soc is True:
    #         print("SOC is not  False, please change SOC")
    #         sys.exit()

    #     return None







