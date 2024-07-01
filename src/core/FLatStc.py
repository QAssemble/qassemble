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
import h5py
import copy
from .Crystal import Crystal
# from .FLatDyn import SigmaGWC
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE

class FLatStc(object):

    def __init__(self,crystal : Crystal):

        self.crystal = crystal

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for js in range(ns):
                matinv[:,:,js,irk] = np.linalg.inv(mat[:,:,js,irk])
        
        return matinv
    
    def K2R(self, matk : np.ndarray = None, rkgrid : list = None)->np.ndarray:

        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matk.shape[0]
        ns = matk.shape[2]
        nrk = matk.shape[3]

        tempmat = copy.deepcopy(matk)
        
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]

                        phase = np.exp(2.0j*np.pi*np.dot(rkvec[irk],delta))
                        
                        # matk[iorb,jorb,js,irk] *= phase
                        tempmat[iorb,jorb,js,irk] *= phase
                        
        
        
        matr = DiagE.fourier.flatstc_k2r(rkgrid,tempmat)

        return matr
    
    def R2K(self,matr : np.ndarray  = None, rkgrid : list = None)->np.ndarray:

        if rkgrid == None:
            rkgrid = self.crystal.rkgrid
        rkvec = self.crystal.kpoint

        norb = matr.shape[0]
        ns = matr.shape[2]
        nrk = matr.shape[3]

        matk = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        matk = DiagE.fourier.flatstc_r2k(rkgrid,matr)
        
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)

                        delta = self.crystal.basisf[a,:] - self.crystal.basisf[b,:]
                        phase = np.exp(-2.0j*np.pi*np.dot(rkvec[irk],delta))

                        matk[iorb,jorb,js,irk] = matk[iorb,jorb,js,irk] * phase
                        
        
        return matk
    
    def Band(self, energy : np.ndarray, fn : str = None, plotoption : bool = False, label : list = None):

        norb = energy.shape[0]
        ns = energy.shape[2]
        nk = energy.shape[3]

        energyplot = np.zeros((norb,ns,nk),dtype=float)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    energyplot[iorb,js,ik] = energy[iorb,iorb,js,ik]
        if plotoption:
            if self.crystal.ns == 1:
                fig,ax = plt.subplots()
                ax.set_xlim(self.crystal.knode[0],self.crystal.knode[-1])
                ax.set_xticks(self.crystal.knode)
                if label == None:
                    pass
                else:
                    ax.set_xticklabels(label)
                for i in range(len(self.crystal.knode)):
                    ax.axvline(x=self.crystal.knode[i],linewidth=0.5,color='r',linestyle='--')
                for iorb in range(norb):
                    ax.plot(self.crystal.kdist,energyplot[iorb,0,:].T,'k-')
                ax.set_ylabel('E (eV)')
                ax.set_title('Band')
                # plt.plot(energyplot.T[:,0,:])
                if fn == None:
                    plt.show()
                else:
                    plt.savefig(fn)
            else:
                up = energyplot[:,0,:]
                down = energyplot[:,1,:]
                plt.plot(up,'k-')
                plt.plot(down,'r-')
                if fn == None:
                    plt.show()
                else:
                    plt.savefig(fn)
        else:
            with open('band.dat','w') as f:
                for js in range(ns):
                    for ik in range(nk):
                        linedata = [self.crystal.kdist[ik]]+energyplot[:,js,ik].tolist()
                        line = ' '.join(map(str,linedata))
                        f.write(line+'\n')

        
        return None
    
    def Diagonalize(self,matk : np.ndarray, eigvec : bool = False):
        
        nk = matk.shape[3]
        norb = matk.shape[0]
        ns = matk.shape[2]
        
        energy = np.zeros((norb,norb,ns,nk),dtype=float)
        evec = np.zeros((norb,norb,ns,nk),dtype=np.complex64)

        # if eigvec == False:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             e = np.linalg.eigvalsh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #     return energy
        # else:
        #     for ik in range(nk):
        #         for js in range(ns):
        #             (e,v) = np.linalg.eigh(matk[:,:,js,ik])
        #             energy[:,:,js,ik] = np.diag(e)
        #             evec[:,:,js,ik] = v

        #     return energy, evec
        if eigvec == False:
            for ik in range(nk):
                for js in range(ns):
                    e,v,info = scipy.linalg.lapack.zheev(matk[:,:,js,ik])
                    energy[:,:,js,ik] = np.diag(e)
            return energy
        else:
            for ik in range(nk):
                for js in range(ns):
                    e,v,info = scipy.linalg.lapack.zheev(matk[:,:,js,ik])
                    energy[:,:,js,ik] = np.diag(e)
                    evec[:,:,js,ik] = v

            return energy, evec
    
    def Gaussian(self, x, mu, sigma = 0.1):

        return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        
    def DOS(self,hamr : np.ndarray = None, sigma : float = 0.1, kgrid : list = [20,20,20], plotoption : bool = False, emax : float = 10, emin : float = -10):


        print("***** DOS Calculation Start *****")
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        if type(kgrid)==list:
            nk = kgrid[0]*kgrid[1]*kgrid[2]
            kpointtemp = np.array(list(itertools.product(np.linspace(0,1,num=kgrid[2],endpoint=False),np.linspace(0,1,num=kgrid[1],endpoint=False),np.linspace(0,1,num=kgrid[0],endpoint=False))))
            kpoint = np.fliplr(kpointtemp)
        elif type(kgrid)==np.ndarray:
            nk = len(kgrid)
            kpoint = kgrid

        print("***** Fourier transfrom R2K Start")
        hamk = self.R2KArb(hamr,kpoint)
        print("***** Fourier transfrom R2K Finish")
        print("***** Hamiltonian Diagonalization Start *****")
        (energy,eigvec) = self.Diagonalize(matk=hamk,eigvec=True)
        print("***** Hamiltonian Diagonalization Finish *****")
        emin = emin#energy[0,0,0].min()
        emax = emax#energy[-1,-1,0].max()
        energyrange=np.linspace(emin,emax,nk)
        # dos = np.zeros_like(energyrange)
        dos = np.zeros((norb,ns,nk),dtype=float)
        tempmat = np.zeros((norb,ns,nk),dtype=float)

        print("***** Gaussian Approach Start *****")
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    e = energy[iorb,iorb,js,ik]
                    tempmat[iorb,js] += self.Gaussian(energyrange,e,sigma)/nk
        print("***** Gaussian Approach Finish *****")
        
        for ik in range(nk):
            for js in range(ns):
                tempmat2 = np.linalg.inv(eigvec[:,:,js,ik])
                # tempmat3 = np.array(np.dot(tempmat2,eigvec[:,:,js,ik]),dtype=float)
                D = np.diag(tempmat[:,js,ik])
                tempmat3 = eigvec[:,:,js,ik]@(D@tempmat2)
                for iorb in range(norb):
                    dos[iorb,js,ik] = tempmat3[iorb,iorb]
                # for jorb in range(norb):
                #     for iorb in range(norb):
                #         # dos[iorb,js,ik] = tempmat2[iorb,jorb]*tempmat[jorb,js,ik]*eigvec[jorb,iorb,js,ik]
                #         # dos[iorb,js,ik] = tempmat3[iorb,jorb]*tempmat[jorb,js,ik]
                #         dos[iorb,js,ik] = eigvec[jorb,iorb,js,ik]*tempmat[jorb,js,ik]*tempmat2[jorb,iorb]
        

        print(f"Integration gaussian : {np.trapz(self.Gaussian(energyrange,0),energyrange)}")
        temp = 0
        for js in range(ns):
            for iorb in range(norb):
                temp+= np.trapz(dos[iorb,js],energyrange)

        
        print(f'Integration dos : {temp}')
        if plotoption:
            fig, ax = plt.subplots()
            ax.set_xlim(energyrange[0],energyrange[-1])
            legend = []
            for js in range(ns):
                for iorb in range(norb):
                    ax.plot(energyrange,dos[iorb,js])
                    legend.append(iorb+1)
            ax.legend(legend)
            ax.set_xlabel('E (eV)')
            ax.set_ylabel('DOS')
            plt.show()
        else:
            with open('dos.dat','w') as f:
                for i in range(len(energyrange)):
                    f.write(f'{energyrange[i]}  {dos[i]}')
        print("***** DOS Calculation Finish *****")
        return None
        
        

    def Visualization(self, energy : np.ndarray, fn : str = None):

        if self.crystal.rkgrid[2] != 1:
            print("Energy surface for only 2D case")
            sys.exit()
        else:
            norb = energy.shape[0]
            ns = energy.shape[2]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            kx = self.crystal.kpoint[:,0].reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2])
            ky = self.crystal.kpoint[:,1].reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2])
            energy = energy.T
            energy = energy.reshape(self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2],ns,norb,norb)

            for js in range(ns):
                for iorb in range(norb):
                    ax.plot_surface(kx[:,:,0],ky[:,:,0],energy[:,:,0,js,iorb,iorb])

            ax.view_init(azim=-120,elev=0)
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Energy eV')
            if fn is None:
                plt.show()
            elif fn is not None:
                fig.savefig(fn)
        
        return None

    def Mixing(self, iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray) -> np.ndarray:

        #norb = Fb.shape[0]
        #ns = Fb.shape[2]
        #nrk = Fb.shape[3]
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)


        Fnew = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        # print(Fnew.shape)
        if iter == 1:
            mix = 1.0
            Fm = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        Fnew[iorb,jorb,js,irk] = mix*Fb[iorb,jorb,js,irk] + (1.0-mix)*Fm[iorb,jorb,js,irk]

        return Fnew
    
    def ChemEmbedding(self,mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        chem = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    chem[iorb,iorb,js,irk] = mu

        return chem

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        matout = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        matout = DiagE.dyson.flatstc(mat1,mat2)

        return matout

    def Projection(self, matin : np.ndarray):

        norb = len(self.crystal.fin)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex64,order='F')

        for ispace in range(nspace):
            matout[...,ispace] = DiagE.projection.flatstc(matin,self.crystal.fprojector[...,ispace])

        return matout
    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[3]

        # if os.path.exists('flatstc'):
        #     pass
        # else:
        #     os.mkdir("flatstc")
        # os.chdir("flatstc")
        with open(fn+'.txt','w') as f:
            f.write("#iorb, jorb, is, ik, Re(F(k)), Im(F(k))\n")
            for irk in range(nrk):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {irk} {matin[iorb,jorb,js,irk].real} {matin[iorb,jorb,js,irk].imag}\n")
        # os.chdir("..")
        return None
    
    def R2KArb(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

        # if self.crystal.kpath == None:
        #     print("Error, kpath doesn't generate")
        #     sys.exit()
        # kpoint = self.crystal.kpath
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nr = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]
        nk = len(kpoint)

        self.crystal.Rvec()
        tempmat = copy.deepcopy(matr)
        matk = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        temp = 0
                        for ir in range(nr):
                            temp += tempmat[iorb,jorb,js,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@self.crystal.rvec[ir]))
                        [a,m1] = self.crystal.FAtomOrb(iorb)
                        [b,m2] = self.crystal.FAtomOrb(jorb)
                        delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                        phase = np.exp(-2.0j*np.pi*(kpoint[ik]@delta))
                        matk[iorb,jorb,js,ik] = temp*phase
        
        return matk

    def HermitianCheck(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = self.crystal.rkgrid[0]*self.crystal.rkgrid[1]*self.crystal.rkgrid[2]


        errmessage = 'The matrix is not hermitian. Check the input file again'
        for ik in range(nk):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        err = matin[iorb,jorb,js,ik]-np.conjugate(matin[jorb,iorb,js,ik])
                        if abs(err)>1.0e-6:
                            print(errmessage)
                            sys.exit()
        return None
    
    def CheckGroup(self, filepath :str, group : str):
        
        with h5py.File(filepath,'r') as file:
            return group in file
    
class NIHamiltonian(FLatStc):

    def __init__(self, crystal: Crystal = None,hoppinglist : list=None, onsitelist : list=None):
        super().__init__(crystal)
        self.hoppinglist = hoppinglist
        self.onsitelist = onsitelist
        print(self.onsitelist)
        self.k = None
        self.r = None
        # self.Hopping()
        # self.Onsite()

        self.Cal()

    def Cal(self): #GenHam
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        kvec = self.crystal.kpoint

        hamtb = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb,ns,self.crystal.rkgrid[0],self.crystal.rkgrid[1],self.crystal.rkgrid[2]),dtype=np.complex64,order='F')
        
        
        for js in range(ns):
            for hopp in self.hoppinglist:
                tij = hopp[0]
                # iorb = hopp[1]
                # jorb = hopp[2]
                (a,m) = hopp[1]
                (b,mp) = hopp[2]
                iorb = self.crystal.FIndex([a,m])
                jorb = self.crystal.FIndex([b,mp])
                R = hopp[3]
                
                # tempmat[iorb,jorb,js,R[0],R[1],R[2]] += -tij
                if (iorb==jorb)and(R==[0,0,0]):
                    print("Wrong value entered, please check the input.ini file")
                    sys.exit()
                else:
                    tempmat[iorb,jorb,js,R[0],R[1],R[2]] += -tij
                    tempmat[jorb,iorb,js,-R[0],-R[1],-R[2]] += -tij.conjugate()

                # 0 == -0

        if self.onsitelist != None:
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb,iorb,js,0,0,0] = +self.onsitelist[iorb]
        # Hermitian check
        tempmat = tempmat.reshape((norb,norb,ns,nk),order='F')
        self.r = tempmat
        hamtb = self.R2K(tempmat)
        self.HermitianCheck(hamtb)

        self.k = hamtb

        return None
    
    def Save(self):
        
        # if os.path.exists('niham'):
        #     pass
        # else:
        #     os.mkdir('niham')
        # os.chdir('niham')
        os.chdir('work')
        
        filepath = 'flatstc.h5'
        groupname = 'niham'
        with h5py.File(filepath,'a') as file:
            if self.CheckGroup(filepath,groupname):
                group = file[groupname]
            else:
                group=file.create_group(groupname)
            
            group.create_dataset('h0k',dtype=complex,data=self.k)
        os.chdir('..')
        return None

    # def Hopping(self):
    #     pass
    
    # def Onsite(self):
    #     pass

class SigmaHartree(FLatStc):

    def __init__(self, crystal: Crystal, occ = None , vbare :np.ndarray = None, onsite : np.ndarray = None): # green -> occ
        super().__init__(crystal)
        self.r = None
        self.k = None
        self.hdyn = None
        self.vbare = vbare
        self.onsiter = onsite
        self.occ = occ
        
        self.Cal()
        # self.MakeDyn()
    
    def Cal(self):
        # vbare = self.vbare.k
        occ = self.occ
        # vk = self.vbare.Double2Quad(self.vbare.k)
        norbc = len(self.crystal.find) #occk.shape[0]
        ns = self.crystal.ns#occk.shape[2]
        nk = len(self.crystal.kpoint) #occk.shape[3]
        norb = len(self.crystal.bind) #vbare.shape[0]

        # onsite = self.R2K(self.onsiter)
        h = np.zeros((norbc,norbc,ns,nk),dtype=np.complex64,order='F')

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
            for ik in range(nk):
                for ind1 in range(norb*ns):
                    nn1 = [0]*2
                    ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                    [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                    iorbc1 = self.crystal.FIndex([a,m1])
                    iorbc2 = self.crystal.FIndex([a,m2])
                    for ind2 in range(norb*ns):
                        nn2 = [0]*2
                        ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                        [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                        iorbc3 = self.crystal.FIndex([b,m3])
                        iorbc4 = self.crystal.FIndex([b,m4])
                        # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]
                        h[iorbc1,iorbc2,js,ik] = self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]
            
        else:
            if(self.crystal.soc == True):
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
                for ik in range(nk):
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc2 = self.crystal.FIndex([a,m2])
                        for ind2 in range(norb*ns):
                            nn2 = [0]*2
                            ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                            [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b,m3])
                            iorbc4 = self.crystal.FIndex([b,m4])
                            h[iorbc1,iorbc2,js,ik] = self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                            
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
                for ik in range(nk):
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m2]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc2 = self.crystal.FIndex([a,m2])
                        for ind2 in range(norb*ns):
                            nn2 = [0]*2
                            ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                            [b,[m3,m4]] = self.crystal.BAtomOrb(jorb)
                            iorbc3 = self.crystal.FIndex([b,m3])
                            iorbc4 = self.crystal.FIndex([b,m4])
                            # h[iorbc1,iorbc2,js,ik] += vk[iorbc1,iorbc3,iorbc4,iorbc2,js,ks,0]*occ[iorbc4,iorbc3,ks]*C
                            h[iorbc1,iorbc2,js,ik] += self.vbare[iorb,jorb,js,ks,0]*occ[iorbc4,iorbc3,ks]*C

        self.k = h #+onsite
        self.r = self.K2R(h)

        return None
    
    def Save(self, fn: str):
        
        os.chdir('work')
        
        filepath = 'flatstc.h5'
        groupname = 'sigmah'
        with h5py.File(filepath,'a') as file:
            if self.CheckGroup(filepath,groupname):
                group = file[groupname]
            else:
                group=file.create_group(groupname)
            
            group.create_dataset(fn,dtype=complex,data=self.k)
        os.chdir('..')
        return None
    
    # def MakeDyn(self):

    #     norb = self.green.gkf.shape[0]
    #     ns = self.green.gkf.shape[2]
    #     nrk = self.green.gkf.shape[3]
    #     nft = self.green.gkf.shape[4]

    #     tempmat = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex64,order='F')

    #     for ift in range(nft):
    #         tempmat[...,ift] = self.hk
    #     self.hdyn = tempmat

    #     return 


class SigmaFock(FLatStc):

    def __init__(self, crystal: Crystal,occr = None, vbare : np.ndarray = None): # green -> occ
        super().__init__(crystal)
        self.r = None
        self.k = None
        self.fdyn = None
        # self.green = green
        self.occr = occr
        self.vbare = vbare
        
        self.Cal()
        # self.MakeDyn()

    def Cal(self):
        
        # g0rt = self.green.glatrt
        occr = self.occr
        # vr = self.vbare.Double2Quad(self.vbare.r)
        
        norbc = len(self.crystal.find)
        ns = occr.shape[2]
        nr = occr.shape[3]
        norb = len(self.crystal.bind)

        fr = np.zeros((norbc,norbc,ns,nr),dtype=np.complex64,order='F')

        # for ir in range(nr):
        #     for js in range(ns):
        #         for iorb in range(norb):
        #             [iorbc1,iorbc4] = self.crystal.b2f[iorb]
        #             for jorb in range(norb):
        #                 [iorbc2,iorbc3] = self.crystal.b2f[jorb]
        #                 fr[iorbc1,iorbc3,js,ir] = -occr[iorbc4,iorbc2,js,ir]*vr[iorb,jorb,js,js,ir]
        for ir in range(nr):
            for ind1 in range(norb*ns):
                nn1 = [0]*2
                ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                iorbc1 = self.crystal.FIndex([a,m1])
                iorbc4 = self.crystal.FIndex([a,m4])
                for ind2 in range(norb*ns):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                    [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b,m3])
                    iorbc2 = self.crystal.FIndex([b,m2])
                    if js == ks:
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]
                        fr[iorbc1,iorbc2,js,ir] += -occr[iorbc4,iorbc3,js,ir]*self.vbare[iorb,jorb,js,ks,ir]
                        
                        # fr[iorbc1,iorbc2,js,ir] += -occr[iorbc3,iorbc4,js,ir]*vr[iorbc1,iorbc3,iorbc2,iorbc4,js,ks,ir]

        fk = self.R2K(fr)

        self.r = fr
        self.k = fk
        del fr, occr
        return None
    
    def Save(self, fn: str):
        
        os.chdir('work')
        
        filepath = 'flatstc.h5'
        groupname = 'sigmaf'
        with h5py.File(filepath,'a') as file:
            if self.CheckGroup(filepath,groupname):
                group = file[groupname]
            else:
                group=file.create_group(groupname)
            
            group.create_dataset(fn,dtype=complex,data=self.k)
        os.chdir('..')

        return None
    
class Hamiltonian(FLatStc):

    def __init__(self, crystal: Crystal, ham : np.ndarray, beta : float = None, sigmah :SigmaHartree = None, sigmaf : SigmaFock = None, sigmac : object = None):
        super().__init__(crystal)

        self.occ = None
        self.occk = None
        self.occr = None
        self.ham = ham
        self.sigmah = sigmah
        self.sigmaf = sigmaf
        self.sigmac = sigmac
        self.beta = beta
        self.k = None
        self.r = None
        self.kmu0 = None
        self.mu = 0
        # self.muold = mu
        self.CalMu0()
        self.SearchMu()

    def CalMu0(self) -> np.ndarray:
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)

        tempmat = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        
        tempmat = copy.deepcopy(self.ham)
        

        if (self.sigmah != None):
            tempmat += self.sigmah.k
        
        if (self.sigmaf != None):
            tempmat += self.sigmaf.k
        
        if (self.sigmac != None):
            z = self.sigmac.z
            sigma = self.sigmac.stck
            # chem = np.zeros((norb,norb,ns,nk),dtype=np.complex64,order='F')
            tempmat2 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat3 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat4 = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
            tempmat4 = copy.deepcopy(tempmat)
            eigval, eigvec = self.Diagonalize(z,True)
            for ik in range(nrk):
                for js in range(ns):
                    for iorb in range(norb):
                        # chem[iorb,iorb,js,ik] = -self.mu
                        if 0<=(eigval[iorb,iorb,js,ik])<=1:
                            continue
                        else:
                            print("Error : The z-factor was calculated incorrectly. Please rerun the code.")
                            print(eigval[iorb,iorb,js,ik])
                            sys.exit()
                    tempmat2[:,:,js,ik] = np.dot(np.dot(eigvec[:,:,js,ik],np.sqrt(eigval[:,:,js,ik])),np.linalg.inv(eigvec[:,:,js,ik]))
            
            tempmat4 = tempmat4 + sigma

            for ik in range(nrk):
                for js in range(ns):
                    tempmat3[:,:,js,ik] = np.dot(np.dot(tempmat2[:,:,js,ik],tempmat4[:,:,js,ik]),tempmat2[:,:,js,ik])

            tempmat = copy.deepcopy(tempmat3)
            del tempmat2, tempmat3, tempmat4

        self.hkmu0 = copy.deepcopy(tempmat)
        del tempmat
        return None

    def NumOfE(self,  mu : float) -> np.ndarray:

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nk = len(self.crystal.kpoint)
        

        energy = self.Diagonalize(self.hkmu0)

        Ne = 0

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    Ne += 1/(1+np.exp((energy[iorb,iorb,js,ik]-mu)*self.beta))

        Ne /= nk        
        N = self.crystal.nume
        

        return N - Ne
    
    def SearchMu(self):
        
        energy = self.Diagonalize(self.hkmu0)
        norb = energy.shape[0]
        mumin = energy[0,0].min()-1000
        mumax = energy[norb-1,norb-1].max()+1000

        nmin = self.NumOfE(mumin)
        nmax = self.NumOfE(mumax)
        if (nmin < 0) or (nmax>0):
            print("Chemical potential is out of the bisection range")
            sys.exit()
        sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)
        # try:
        #     sol = scipy.optimize.brentq(self.NumOfE,mumin,mumax)
        # except:
        #     sol = scipy.optimize.newton(self.NumOfE,0,tol=10**(-10))
        self.mu = sol
        
        self.UpdateMu()
        return None


    def Occ(self) -> np.ndarray:
        
        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        
        # energy = self.Diagonalize(self.hk)

        occk = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')
        occ = np.zeros((norb,norb,ns),dtype=np.complex64,order='F')
        tempmat = np.zeros((norb,norb),dtype=float,order='F')

        energy, eigvec = self.Diagonalize(self.k,True)
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    tempmat[iorb,iorb] = 1/(np.exp(energy[iorb,iorb,js,irk]*self.beta)+1)
                # occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,np.linalg.inv(eigvec[:,:,js,irk])))
                occk[:,:,js,irk] = np.dot(eigvec[:,:,js,irk],np.dot(tempmat,scipy.linalg.inv(eigvec[:,:,js,irk])))
                
            occ += occk[...,irk]
        
        occ /= nrk
        
        self.occ = occ
        self.occk = occk
        self.occr = self.K2R(occk)

        return None
    
    def UpdateMu(self) -> np.ndarray:

        chem = self.ChemEmbedding(self.mu)

        ham = self.hkmu0 - chem
        hamr = self.K2R(ham)
        self.k = ham
        self.r = hamr
        self.Occ()

        return None
    
    def Save(self, fn: str):
        os.chdir('work')
        
        filepath = 'flatstc.h5'
        groupname = 'sigmah'
        with h5py.File(filepath,'a') as file:
            if self.CheckGroup(filepath,groupname):
                group = file[groupname]
            else:
                group=file.create_group(groupname)
            
            group.create_dataset(fn,dtype=complex,data=self.k)
        os.chdir('..')
        return None
