import string as string
import matplotlib as mat
import re as re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
import json, os, shutil
import itertools
import scipy.optimize
from sympy.physics.wigner import gaunt, wigner_3j
from scipy.fftpack import fftn, ifftn
import scipy.linalg
import sys
import subprocess
from Common import *
diage_path = os.environ.get('DIAGE','')
path = diage_path+"/modules"
sys.path.append(path)
import DiagE



class Crystal():
    def __init__(self,latt : list,basis_position : list):
        latt = np.array(latt,dtype=float)
        basis_position = np.array(basis_position,dtype=float)
        self.latt = latt
        self.basis_f = basis_position
        self.basis_c = None
        self.bvec = np.zeros((3,3))
        self.vol=np.dot(np.cross(latt[:,0], latt[:,1]), latt[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(latt[:,1], latt[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(latt[:,2], latt[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(latt[:,0], latt[:,1])/self.vol
        self.kpoint = None
        self.kpath = None
        self.grid = []

    def Cartesian_Basis(self):
        self.basis_c = np.dot(self.basis_f,self.lat)


    def Kpoint(self,fname=None,meshgrid=None,karray=None):
        if (fname is not None):
            kpoint=np.load(fname)
            nk = len(kpoint)
            self.kpoint = kpoint
            self.nk = nk
        elif (meshgrid is not None):
            meshgrid = np.array(meshgrid)
            nk = meshgrid[0]*meshgrid[1]*meshgrid[2]
            kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=meshgrid[2],endpoint=False),np.linspace(0,1,num=meshgrid[1],endpoint=False),np.linspace(0,1,num=meshgrid[0],endpoint=False))))
            kpoint=np.fliplr(kpoint_temp)
            self.kpoint = kpoint
            self.grid = meshgrid
            self.nk = nk
        elif (karray is not None):
            kpoint = karray
            nk = len(kpoint)
            self.kpoint = kpoint
            self.nk = nk

    def k_path(self,kpath : list =None,nk : int = None) -> np.ndarray:


        kpath = np.array(kpath,dtype=float)
        nnod = kpath.shape[0]
        k_mat = np.linalg.inv(np.dot(self.latt,self.latt.T))
        knode = np.zeros(nnod,dtype=float)
        for n in range(1,nnod):
            dk = kpath[n] - kpath[n-1]
            l = np.sqrt(np.dot(dk,np.dot(k_mat,dk)))
            knode[n] = knode[n-1]+l

        nk = nk*(nnod-1)

        ind_nod = [0]
        for n in range(1,nnod-1):
            frac = knode[n]/knode[-1]
            ind_nod.append(int(round(frac*(nk-1))))
        ind_nod.append(nk-1)


        k_vec = np.zeros((nk,kpath.shape[1]))
        k_vec[0] = kpath[0]
        cnt = 0
        for i in range(1,nnod):
            n_i = ind_nod[i-1]
            n_f = ind_nod[i]
            k_i = kpath[i-1]
            k_f = kpath[i]
            for j in range(n_i,n_f+1):
                frac = float(j-n_i)/float(n_f-n_i)
                k_vec[j] = k_i + frac*(k_f-k_i)

        self.kpath = k_vec

class FHamiltonian(object):

    def __init__(self,crystal : Crystal, ns : int = None, SOC : bool = False):

        self.ns = ns
        self.basis_f = crystal.basis_f
        self.basis_c = crystal.basis_c
        self.latt = crystal.latt
        self.bvec = crystal.bvec
        self.kpoint = crystal.kpoint
        self.kpath = crystal.kpath
        self.vol = crystal.vol
        self.rkgrid = crystal.grid
        self.SOC = SOC
        self.find = {}
        self.Hopping = []
        self.rvec = []
        self.Onsite = []
        self.Ham_R = None
    
    def set_basis_index(self,option : list = None)->dict:

        ind = []
        for m1 in range(option[1]):
            ind.append([option[0],m1])
        
        norbc = len(self.find)
        ii = 0
        for iorbc in range(norbc,norbc+option[1]):
            self.find[iorbc] = ind[ii]
            ii +=1
        norbc = len(self.find)

        self.Ham_R = np.zeros((norbc,norbc),dtype=complex,order='F')

    def Fatomorb(self,key : int = None) -> list:
        '''
        input : composite index for fermion
        output : atom and orbital index in fermion case

        e.g.
        0 -> [0,0]
        '''
        return self.find[key]
    
    def Findex(self,val : list = None) -> int:
        '''
        input : atom and orbital index with list
        output : composite index for fermion

        e.g.
        [0,0] -> 0
        '''
        
        for key, value in self.find.items():
            if value == val:
                return key
    
    def Hoppinglist(self,hopping : float = 0, ind_i : int = 0, ind_j : int = 0, R : list = [])->list:

        R = np.array(R)
        self.Ham_R[ind_i,ind_j] = hopping
        self.Ham_R[ind_j,ind_i] = hopping

        alpha = self.Fatomorb(ind_i)[0]
        beta = self.Fatomorb(ind_j)[0]

        rv = self.basis_f[alpha,:] - self.basis_f[beta,:] + R

        self.rvec.append([ind_i, ind_j, rv])

        self.Hopping.append([hopping,ind_i,ind_j,R])

    def On_site_list(self,Energy : list = []) -> list:

        self.Onsite.append(Energy)

        for iorb, e in enumerate(Energy):
            self.Ham_R[iorb,iorb] = e

    def Hamiltonian(self, flag : int = 0) -> np.ndarray:

        if flag == 0:
            kvec = self.kpoint
        elif flag == 1:
            kvec = self.kpath
        else:
            print("flag has onle 0 or 1")
            sys.exit()
        
        nk = len(kvec)
        norb = len(self.find)
        ns = self.ns
        Hmat = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        
        for js in range(ns):
            for iorb in range(norb):
                Hmat[iorb,iorb,js,:] = self.Ham_R[iorb,iorb]

        for hopp in self.Hopping:
            t = hopp[0]
            iorb = hopp[1]
            jorb = hopp[2]
            R = hopp[3]
            [a,m1] = self.Fatomorb(iorb)
            [b,m2] = self.Fatomorb(jorb)
            
            rvec = self.basis_f[a,:]-self.basis_f[b,:]+ R
            phase = np.exp(-2.0j*np.pi*np.dot(kvec,rvec))
            for js in range(ns):
                for ik in range(nk):
                    Hmat[iorb,jorb,js,ik] += t*phase[ik]
                    Hmat[jorb,iorb,js,ik] += t*phase[ik].conjugate()

        self.Ham_tb = Hmat
        
    
    def diagonalize(self,Hmat : np.ndarray, eigvec : bool = False):

        nk = Hmat.shape[3]
        norb = Hmat.shape[0]
        ns = self.ns
        
        Energy = np.zeros((norb,norb,ns,nk),dtype=float)
        evec = np.zeros((norb,norb,ns,nk),dtype=complex)

        if eigvec == False:
            for ik in range(nk):
                for js in range(ns):
                    e = np.linalg.eigvalsh(Hmat[:,:,js,ik])
                    Energy[:,:,js,ik] = np.diag(e)
            return Energy
        else:
            for ik in range(nk):
                for js in range(ns):
                    (e,v) = np.linalg.eig(Hmat[:,:,js,ik])
                    Energy[:,:,js,ik] = np.diag(e)
                    evec[:,:,js,ik] = v
            return Energy, evec
    
    def visualization(self,energy : np.ndarray, filename : str = None):
        '''
        For test the code
        '''

        if self.rkgrid[2]==1:
            norb = energy.shape[0]
            ns = self.ns
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            kx = self.kpoint[:,0].reshape(self.rkgrid[0],self.rkgrid[1],self.rkgrid[2])
            ky = self.kpoint[:,1].reshape(self.rkgrid[0],self.rkgrid[1],self.rkgrid[2])

            energy = energy.T
            energy = energy.reshape(self.rkgrid[0],self.rkgrid[1],self.rkgrid[2],ns,norb,norb)

            for js in range(ns):
                for iorb in range(norb):
                    ax.plot_surface(kx[:,:,0],ky[:,:,0],energy[:,:,0,js,iorb,iorb])
            
            ax.view_init(azim=-120,elev=0)
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Energy eV')
            plt.show()
            if filename is not None:
                fig.savefig(filename)
        
        elif self.rkgrid[2] != 1:
            print('Error, kz must be 1')
            sys.exit()

    def band(self, energy : np.ndarray):

        import matplotlib.pyplot as plt

        norb = energy.shape[0]
        ns = self.ns
        nk = energy.shape[3]

        energy_plot = np.zeros((norb,ns,nk),dtype=complex)

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    energy_plot[iorb,js,ik] = energy[iorb,iorb,js,ik]

        if ns == 1:
            plt.plot(energy_plot.T[:,0,:])
            plt.show()
        elif ns == 2:
            
            up = energy_plot.T[:,0,:]
            down = energy_plot.T[:,1,:]

            plt.plot(up,'k-')
            plt.plot(down,'r-')
            plt.show()
    
    def FInv(self,mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]

        mat_inv = np.zeros((norb,norb),dtype=complex,order='F')

        mat_inv = np.linalg.inv(mat)

        return mat_inv
    
    def FInvLocStc(self,mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]

        mat_inv = np.zeros((norb,norb,ns),dtype=complex,order='F')

        for js in range(ns):
            mat_inv[:,:,js] = self.FInv(mat[:,:,js])
        
        return mat_inv
    
    def FInvLatStc(self,mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nk = mat.shape[3]

        mat_inv = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            mat_inv[:,:,:,ik] = self.FInvLocStc(mat[:,:,:,ik])

        return mat_inv
    
    def FInvLocDyn(self,mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nt = mat.shape[3]

        mat_inv = np.zeros((norb,norb,ns,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_inv[:,:,:,it] = self.FInvLocStc(mat[:,:,:,it])

        return mat_inv
    
    def FInvLatDyn(self,mat : np.ndarray) -> np.ndarray:

        norb = mat.shape[0]
        ns = mat.shape[2]
        nk = mat.shape[3]
        nt = mat.shape[4]

        mat_inv = np.zeros((norb,norb,ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_inv[:,:,:,:,it] = self.FInvLatStc(mat[:,:,:,:,it])

        return mat_inv
    
    def FLatStc_K2R(self,rkgrid : list = None, hmatk : np.ndarray = None) -> np.ndarray:

        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nk = rkgrid[0]*rkgrid[1]*rkgrid[2]

        rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
        rkvec = np.fliplr(rkvec)

        norb = hmatk.shape[0]

        for iorb in range(norb):
            for jorb in range(norb):
                [a,m1] = self.Fatomorb(iorb)
                [b,m2] = self.Fatomorb(jorb)

                delta = self.basis_f[a,:]-self.basis_f[b,:]
                phase = np.exp(2.0j*np.pi*np.dot(rkvec,delta))

                for ik in range(nk):
                    hmatk[iorb,jorb,:,ik] *= phase[ik]

        hmatr = DiagE.fourier.flatstc_k2r(rkgrid,hmatk)

        return hmatr

    def FLatStc_R2K(self,rkgrid : list = None, hmatr : np.ndarray = None) -> np.ndarray:


        norb = hmatr.shape[0]
        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nr = rkgrid[0]*rkgrid[1]*rkgrid[2]
        hmatk = DiagE.fourier.flatstc_r2k(rkgrid,hmatr)

        rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
        rkvec = np.fliplr(rkvec)

        for iorb in range(norb):
            for jorb in range(norb):

                [a,m1] = self.Fatomorb(iorb)
                [b,m2] = self.Fatomorb(jorb)

                delta = self.basis_f[a,:]-self.basis_f[b,:]
                phase = np.exp(-(2.0j)*np.pi*np.dot(rkvec,delta))

                for ir in range(nr):
                    hmatk[iorb,jorb,:,ir] *= phase[ir]

        return hmatk

    def FLatDyn_K2R(self,rkgrid : list = None, hmatk : np.ndarray = None) -> np.ndarray :
        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nto = hmatk.shape[4]
        norb = hmatk.shape[0]
        ns = hmatk.shape[2]
        nk = hmatk.shape[3]
        hmatr = np.zeros((norb,norb,ns,nk,nto),dtype=complex,order='F')

        for ito in range(nto):
            hmatr[:,:,:,:,ito] = self.FLatStc_K2R(rkgrid,hmatk[:,:,:,:,ito])

        return hmatr

    def FLatDyn_R2K(self,rkgrid : list = None, hmatr : np.ndarray = None) -> np.ndarray :
        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nto = hmatr.shape[4]
        norb = hmatr.shape[0]
        ns = hmatr.shape[2]
        nk = hmatr.shape[3]
        hmatk = np.zeros((norb,norb,ns,nk,nto),dtype=complex,order='F')

        for ito in range(nto):
            hmatk[:,:,:,:,ito] = self.FLatStc_R2K(rkgrid,hmatr[:,:,:,:,ito])

        return hmatk
    
    def FLocDyn_M(self, omega : np.ndarray = None, ff : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:
        
        norb = ff.shape[0]

        momentum = np.zeros((norb,norb,self.ns,3),dtype=complex,order='F')
        high = np.zeros((norb,norb,self.ns),dtype=complex,order='F')
        momentum, high = DiagE.fourier.flocdyn_m(omega,ff,isgreen,highzero)

        return momentum,high
    
    def FLatDyn_M(self, omega : np.ndarray = None, ff : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:
        
        norb = ff.shape[0]
        nk = ff.shape[3]

        momentum = np.zeros((norb,norb,self.ns,nk,3),dtype=complex,order='F')
        high = np.zeros((norb,norb,self.ns,nk),dtype=complex,order='F')
        momentum, high = DiagE.fourier.flatdyn_m(omega,ff,isgreen,highzero)

        return momentum,high
    
    def FLocDyn_T2F(self,tau : np.ndarray = None, ftau : np.ndarray = None, freq : np.ndarray = None) -> np.ndarray:
        
        ff = np.empty_like(ftau,dtype=complex,order='F')

        ff = DiagE.fourier.flocdyn_t2f(tau,ftau,freq)

        return ff
    
    def FLatDyn_T2F(self,tau : np.ndarray = None, ftau : np.ndarray = None, freq : np.ndarray = None) -> np.ndarray:

        nk = ftau.shape[3]
        ff = np.empty_like(ftau,dtype=complex,order='F')

        ff = DiagE.fourier.flatdyn_t2f(tau,ftau,freq)
        
        return ff
    
    def FLocDyn_F2T(self, omega : np.ndarray = None, ff : np.ndarray = None, tau : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:

        momentum, high = self.FLocDyn_M(omega,ff,isgreen,highzero)

        ftau = np.empty_like(ff,dtype=complex,order='F')
        ftau = DiagE.fourier.flocdyn_f2t(omega,ff,momentum,tau)

        return ftau
    
    def FLatDyn_F2T(self, omega : np.ndarray = None, ff : np.ndarray = None, tau : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:
        
        nk = ff.shape[3]
        momentum, high = self.FLatDyn_M(omega,ff,isgreen,highzero)
        ftau = np.empty_like(ff,dtype=complex,order='F')

        ftau = DiagE.fourier.flatdyn_f2t(omega,ff,momentum,tau)

        return ftau
    
    def FimpconvertLocStc(self,mat : np.ndarray, flag : int):

        norb = mat.shape[0]
        ns = mat.shape[2]
        

        if flag == 1: # problem -> space
            nspace = 0
            for key, val in self.prob_space.items():
                nspace+=len(val)
            mat_ret = np.zeros((norb,norb,ns,nspace),dtype=complex,order='F')
            for key, val in self.prob_space.items():
                iprob = int(key)-1
                for ispace in val:
                    mat_ret[...,ispace] = mat[...,iprob]
            
            return mat_ret
        elif flag == 0: #space -> problem
            nprob = len(self.prob_space)
            mat_ret = np.zeros((norb,norb,ns,nprob),dtype=complex,order='F')
            for key, val in self.prob_space.items():
                iprob = int(key)-1
                tempmat = np.zeros((norb,norb,ns),dtype=complex,order='F')
                for ispace in val:
                    tempmat += mat[...,ispace]
                tempmat /= len(val)
                mat_ret[...,iprob] = tempmat

            return mat_ret


    
    def FimpconvertLocDyn(self,mat : np.ndarray, flag : int):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]
        

        if flag ==1:
            nspace = 0
            for key, val in self.prob_space.items():
                nspace += len(val)
            mat_ret = np.zeros((norb,norb,ns,nft,nspace),dtype=complex,order='F')
            for ift in range(nft):
                mat_ret[:,:,:,ift,:] = self.FimpconvertLocStc(mat[:,:,:,ift,:],flag)
            return mat_ret
        elif flag==0:
            nprob = len(self.prob_space)
            mat_ret = np.zeros((norb,norb,ns,nft,nprob),dtype=complex,order='F')
            for ift in range(nft):
                mat_ret[:,:,:,ift,:] = self.FimpconvertLocStc(mat[:,:,:,ift,:],flag)
            return mat_ret
    
    def FmixLocStc(self,iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray)->np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        F_new = np.zeros((norb,norb,ns),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0
        
        for js in range(ns):
            for iorb in range(norb):
                for jorb in range(norb):
                    F_new[iorb,jorb,js] = mix*Fb[iorb,jorb,js]+(1-mix)*Fm[iorb,jorb,js]
        
        return F_new
    
    def FmixLatStc(self,iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray)->np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nrk = Fb.shape[3]

        F_new = np.zeros((norb,norb,ns,nrk),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0
        for irk in range(nrk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        F_new[iorb,jorb,js,irk] = mix*Fb[iorb,jorb,js,irk]+(1-mix)*Fm[iorb,jorb,js,irk]
        
        return F_new
    
    def FmixLocDyn(self,iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray)->np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nft = Fb.shape[3]

        F_new = np.zeros((norb,norb,ns,nft),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0

        for ift in range(nft):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        F_new[iorb,jorb,js,ift] = mix*Fb[iorb,jorb,js,ift] + (1-mix)*Fm[iorb,jorb,js,ift]

        return F_new
    
    def FmixLatDyn(self,iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray)->np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nrk = Fb.shape[3]
        nft = Fb.shape[4]

        F_new = np.zeros((norb,norb,ns,nrk,nft),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0
        print(mix)
        # for ift in range(nft):
        #     for irk in range(nrk):
        #         for js in range(ns):
        #             for iorb in range(norb):
        #                 for jorb in range(norb):
        #                     F_new[iorb,jorb,js,irk,ift] = mix*Fb[iorb,jorb,js,irk,ift] + (1-mix)*Fm[iorb,jorb,js,irk,ift]
        F_new = mix*Fb + (1-mix)*Fm
        return F_new
    
    def FgaussianLocDyn(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]
        ynew = np.zeros((norb,norb,ns,nft),dtype=complex,order='F')
        w0 = (1.0-3.0*w1)*np.pi*temperature
        width_array = w0+w1*x
        cnt = 0
        for x0 in x:
            if (x0>cutoff+(w0+w1*cutoff)*3.0):
                ynew[...,cnt] = y[...,cnt]
            else:
                if ((x0>3*width_array[cnt])and((x[-1]-x0)>3*width_array[cnt])):
                    dist = 1.0/np.sqrt(2*np.pi)/width_array[cnt]*np.exp(-(x-x0)**2/2.0/width_array[cnt]**2)
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                ynew[iorb,jorb,js,cnt] = sum(dist*y[iorb,jorb,js])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1
        return ynew
    
    def FgaussianLatDyn(self,x,y,w1,temperature,cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nk = y.shape[3]
        nft = y.shape[4]
        ynew = np.zeros((norb,norb,ns,nk,nft),dtype=complex,order='F')
        for ik in range(nk):
            ynew[...,ik,:] = self.FgaussianLocDyn(x,y[...,ik,:],w1,temperature,cutoff)
        return ynew
    
    def write_dict_LocStc(self,equiv : np.ndarray, mat_in : np.ndarray)->dict:
        
        ns = mat_in.shape[2]
        
        Nind = np.amax(equiv)
        norb = mat_in.shape[0]
        mat_dict = {}
        
        for ind in range(Nind):
            mat_dict[ind+1]=[]
            pos = find_positions(equiv,ind+1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    e+=mat_in[ii,jj,js]
                e/=len(pos)
                mat_dict[ind+1].append(e)

        return mat_dict
    
    def write_dict_LocDyn(self,equiv : np.ndarray, mat_in : np.ndarray)->dict:
        
        ns = mat_in.shape[2]
        Nind = np.amax(equiv)
        mat_dict = {}    

        for ind in range(Nind):
            mat_dict[ind+1]=[]
            pos = find_positions(equiv,ind+1)
            for js in range(ns):
                e = 0
                for ii, jj in pos:
                    
                    e+=mat_in[ii,jj,js]
                e/=len(pos)
                mat_dict[ind+1].append(e.tolist())

        return mat_dict
    
    def read_dict_LocStc(self,equiv : np.ndarray, mat_dict : dict)->np.ndarray:

        norb = len(equiv)
        ns = self.ns
        mat_out = np.zeros((norb,norb,ns),dtype=complex,order='F')

        Nind = np.amax(equiv)
        
        for js in range(ns):
            for ind in range(Nind):
                pos = find_positions(equiv,ind+1)
                for ii,jj in pos:
                    mat_out[ii,jj,js] = mat_dict[str(ind+1)]

        return mat_out
    
    def read_dict_LocDyn(self,equiv : np.ndarray, mat_dict : dict)->np.ndarray:
        
        norb = len(equiv)
        ns = self.ns
        nfreq = len(mat_dict["1"])

        mat_out = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')

        Nind = np.amax(equiv)
        
        for js in range(ns):
            for ind in range(Nind):
                pos = find_positions(equiv,ind+1) 
                for ii, jj in pos:
                    mat_out[ii,jj,js] = mat_dict[str(ind+1)]
        
        return mat_out

    
    def mapping_mR_R(self,rkgrid : list = None): # -> fermion
        
        if rkgrid == None:
            rkvec = self.kpoint 
        else:
            rkgrid = np.array(rkgrid,dtype=int,order='F')
            rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
            rkvec = np.fliplr(rkvec)

        mrkvec = np.array(1.0-rkvec,dtype=float)

        for i in range(mrkvec.shape[0]):
            for j in range(mrkvec.shape[1]):
                if mrkvec[i,j] == 1.0:
                    mrkvec[i,j] = 0
        
        mapping_idx = []

        for i in range(rkvec.shape[0]):
            for j in range(mrkvec.shape[0]):
                if(abs(rkvec[i,0]-mrkvec[j,0])<=1.0e-6)and(abs(rkvec[i,1]-mrkvec[j,1])<=1.0e-6)and((abs(rkvec[i,2]-mrkvec[j,2])<=1.0e-6)):
                    mapping_idx.append([i,j])

        self.mapping_idx = mapping_idx
    
    def FLatTau_m(self,flattau : np.ndarray) -> np.ndarray: # -> fermion

        self.mapping_mR_R()

        nk = flattau.shape[3]
        ntau = flattau.shape[4]
        norb = flattau.shape[0]

        flattau_m = np.zeros((norb,norb,self.ns,nk,ntau),dtype=complex,order='F')

        for itau in range(ntau):
            for kp in self.mapping_idx:
                for js in range(self.ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            flattau_m[iorb,jorb,js,kp[0],itau] = -flattau[iorb,jorb,js,kp[1],ntau-itau-1]

        return flattau_m
    
    def num_of_e_tau(self, mu : float, Nt : float, hmat : np.ndarray, tau : np.ndarray):
        
        norb = hmat.shape[0]
        nk = hmat.shape[3]

        chem = np.zeros((norb,norb,self.ns,nk),dtype=float,order='F')
        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                       chem[iorb,iorb,js,ik] = mu
        
        Ham = hmat-chem
        gf = DiagE.bare.flattau(Ham,tau)
        ntau = gf.shape[4]
        Ne = 0
        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                    Ne += -np.real(gf[iorb,iorb,js,ik,ntau-1])
        
        return Nt-Ne/nk
    
    def root_find_for_hf(self, Nt : float, hmat : np.ndarray, tau : np.ndarray):
        mu_min = -40
        mu_max = 40
        sol = scipy.optimize.bisect(self.num_of_e_tau,mu_min, mu_max, args=(Nt,hmat,tau))

        return sol
    
    def num_of_e_freq(self,mu :float, Nt : float, G : np.ndarray,omega : np.ndarray, tau : np.ndarray):

        norb = G.shape[0]
        ns = self.ns
        nk = G.shape[3]
        nomega = G.shape[4]

        G_cal_f = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')
        G_cal_t = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')

        tempmat = self.FInvLatDyn(G)
        
        # for iomega in range(nomega):
        #     for ik in range(nk):
        #         for js in range(self.ns):
        #             for iorb in range(norb):
        #                 tempmat[iorb,iorb,js,ik,iomega] += mu

        # 
        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    for iorb in range(norb):
                        tempmat[iorb,iorb,js,ik,iomega] = -mu
        # for iomega in range(nomega):
        #     for ik in range(nk):
        #         for js in range(ns):
        #             for iorb in range(norb):
        #                 tempmat[iorb,iorb,js,ik,iomega] += mu
        G_cal_f = DiagE.dyson.flatdyn(G,tempmat)
        # G_cal_f = self.FInvLatDyn(tempmat)
        G_cal_t = self.FLatDyn_F2T(omega,G_cal_f,tau,1,1)

        ntau = G_cal_t.shape[4]
        Ne = 0
        
        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                        Ne += -G_cal_t[iorb,iorb,js,ik,ntau-1]
        Ne /= nk

        return np.real(Nt-Ne)
    
    def root_find_for_GW(self,Nt : float, G : np.ndarray, omega : np.ndarray, tau : np.ndarray):
        
        mu_min = -40
        mu_max = 40

        sol = scipy.optimize.bisect(self.num_of_e_freq,mu_min,mu_max,args=(Nt,G,omega,tau))

        return sol
    
    def occupation_matrix(self, flattau : np.ndarray = None):
        '''
        input : G(k,tau)
        output : occupancy matrix(norb,norb,ns)
        '''

        
        norb = flattau.shape[0]
        nk = flattau.shape[3]
        ntau = flattau.shape[4]
        nmat = np.zeros((norb,norb,self.ns,nk),dtype=complex)

        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        nmat[iorb,jorb,js,ik] += -flattau[iorb,jorb,js,ik,ntau-1]
        
        return nmat
    
    def Hartree(self,Gnot : np.ndarray, boson = None) -> np.ndarray:
        '''
        input : Vinput(k,tau), Gnot(k,tau)
        output : Energy(k,tau)

        it save the hartree self-energy in the class Fhamiltonian
        '''
        V = boson.V_bare
        bf_list = boson.b2f
        nk = Gnot.shape[3]
        ntau = Gnot.shape[4]
        norbc = Gnot.shape[0]
        norb = V.shape[0]

        
        # 3**2 + 5**2 -> norb norbc -> 3+5
        tempmat = np.zeros((norb*self.ns,norb*self.ns,nk),dtype=complex,order='F')

        Energy = np.zeros((norbc,norbc,self.ns,nk),dtype=complex)

        if (self.ns != 1):
            for ik in range(nk):
                for s1 in range(self.ns):
                    for iorb in range(norb):
                        nn1 = [iorb,s1]
                        ind1, nn1 = indexing(norb*self.ns,2,[norb,self.ns],1,0,nn1)
                        iorb1, iorb2 = bf_list[iorb]
                        for s2 in range(self.ns):
                            for jorb in range(norb):
                                nn2 = [jorb,s2]
                                ind2, nn2 = indexing(norb*self.ns,2,[norb,self.ns],1,0,nn2)
                                iorb3, iorb4 = bf_list[jorb]
                                Gf_temp = np.zeros((norbc,norbc,self.ns),dtype=complex)
                                for jk in range(nk):
                                    Gf_temp[iorb4,iorb3,s2] += Gnot[iorb4,iorb3,s2,jk,-1]
                                tempmat[ind1,ind2,ik] = V[iorb,jorb,s1,s2,ik]
                                Energy[iorb1,iorb2,s1,ik] += -tempmat[ind1,ind2,0]*1/nk*Gf_temp[iorb4,iorb3,s2]

                    # for iorb in range(norb):
                    #     [a,[m1,m2]] = self.orbkey2val(iorb)
                    #     iorb1 = self.val2key([a,m1])
                    #     iorb2 = self.val2key([a,m2])
                    #     nn1 = [iorb,s1]
                    #     ind1, nn1 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn1)
                    #     if (iorb1==None)or(iorb2==None):
                    #         continue
                    #     for s2 in range(self.ns):
                    #         for jorb in range(norb):
                    #             [b,[m3,m4]] = self.orbkey2val(jorb)
                    #             jorb1 = self.val2key([b,m3])
                    #             jorb2 = self.val2key([b,m4])
                    #             nn2 = [jorb,s2]
                    #             ind2, nn2 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn2)
                    #             if (jorb1==None)or(jorb2==None):
                    #                continue
                    #             Gf_temp = np.zeros((norbc,norbc,self.ns))
                    #             for jk in range(nk):
                    #                 Gf_temp[jorb2,jorb1,s2] += Gnot[jorb2,jorb1,s2,jk,ntau-1]
                    #             tempmat[ind1,ind2,ik] = Vinput[iorb,jorb,s1,s2,ik]
                    #             Energy[iorb1,iorb2,s1,ik] += -tempmat[ind1,ind2,0]*1/nk*Gf_temp[jorb2,jorb1,s2]
                                # Energy[iorb1,iorb2,s1,ik] += -Vinput[iorb,jorb,s1,s2,0]*1/nk*Gf_temp[jorb2,jorb1,s2]
        else:
            if (self.SOC == True):
                C = 1
                for ik in range(nk):
                    for iorb in range(norb):
                        iorb1, iorb2 = bf_list[iorb]
                        for jorb in range(norb):
                            jorb1, jorb2 = bf_list[jorb]
                            Gf_temp = np.zeros((norbc,norbc,1),dtype=complex)
                            for jk in range(nk):
                                Gf_temp[jorb2,jorb1,0] += Gnot[jorb2,jorb1,0,jk,ntau-1]
                            Energy[iorb1,iorb2,0,ik] += -V[iorb,jorb,0,0,0]*1/nk*Gf_temp[jorb2,jorb1,0]*C
            else:
                C = 2
                for ik in range(nk):
                    for iorb in range(norb):
                        iorb1, iorb2 = bf_list[iorb]
                        for jorb in range(norb):
                            jorb1, jorb2 = bf_list[jorb]
                            Gf_temp = np.zeros((norbc,norbc,1),dtype=complex)
                            for jk in range(nk):
                                Gf_temp[jorb2,jorb1,0] += Gnot[jorb2,jorb1,0,jk,ntau-1]
                            Energy[iorb1,iorb2,0,ik] += -V[iorb,jorb,0,0,0]*1/nk*Gf_temp[jorb2,jorb1,0]*C

        self.Sigma_H = Energy
    
    def Exchange(self, Gf : np.ndarray, boson )-> np.ndarray:
            #(self,Gf, Boson : )
        '''
        input : Vinput(R,tau), Gf(k,tau), rkgrid-> optional
        output : self_energy(R)

        it save the fock self-energy(k) in Fhamiltonian 
        '''
        Gf = self.FLatDyn_K2R(self.rkgrid,Gf)
        Vk = boson.V_bare
        Vr = boson.BLatStc_K2R(boson.rkgrid,Vk)
        bf_list = boson.b2f
        nr = Gf.shape[3]
        ntau = Gf.shape[4]
        norb = Vr.shape[0]
        norbc = Gf.shape[0]
        
        

        Energy = np.zeros((norbc,norbc,self.ns,nr),dtype=complex,order='F')

        

        for ir in range(nr):
            for js in range(self.ns):
                for iorb in range(norb):
                    iorb1, iorb2 = bf_list[iorb]
                    for jorb in range(norb):
                        jorb1, jorb2 = bf_list[jorb]
                        Energy[iorb1,jorb1,js,ir] = Gf[iorb2,jorb2,js,ir,ntau-1] * Vr[iorb,jorb,js,js,ir]

        Energy_k = self.FLatStc_R2K(self.rkgrid,Energy)
        self.Sigma_F = Energy_k
        
    
    def Correlated_self_energy(self, Gf : np.ndarray, boson, FT,rkgrid : list = None) -> np.ndarray:
        '''
        input : Wc(R,tau), Gf(k,tau)

        return : Sigma_C(k,freq)

        it save the correlated self-energy(k,f) in Fhamiltonian
        '''
        
        if rkgrid == None:
            rkgrid=self.rkgrid
        else:
            rkgrid = rkgrid
        Gf = self.FLatDyn_K2R(rkgrid,Gf)
        Wc = boson.Wc
        bf_list = boson.b2f
        tau = FT.tau
        omega = FT.omega

        norbc = Gf.shape[0]
        ns = Gf.shape[2]
        nr = Gf.shape[3]
        ntau = Gf.shape[4]
        norb = Wc.shape[0]

        Energy = np.zeros((norbc,norbc,ns,nr,ntau),dtype=complex,order='F')

        for itau in range(ntau):
            for ir in range(nr):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            iorb1, iorb2 = bf_list[iorb]
                            for jorb in range(norb):
                                jorb1, jorb2 = bf_list[jorb]
                                if js == ks:
                                    Energy[iorb1,jorb1,js,ir,itau] += Gf[iorb2,jorb2,js,ir,itau]*Wc[iorb,jorb,js,ks,ir,itau]
        
        Energy_kt = self.FLatDyn_R2K(self.rkgrid,Energy)
        Energy_kf = self.FLatDyn_T2F(tau,Energy_kt,omega)


        self.Sigma_C = Energy_kf
    
    def Combine_self_energy(self):
        norb = self.Sigma_C.shape[0]
        ns = self.Sigma_C.shape[2]
        nk = self.Sigma_C.shape[3]
        nfreq = self.Sigma_C.shape[4]

        Sigma = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')
        for ifreq in range(nfreq):
            Sigma[:,:,:,:,ifreq] = self.Sigma_C[:,:,:,:,ifreq] + self.Sigma_H + self.Sigma_H
        
        self.Sigma = Sigma
        return None
    
    def int_FLatFreq(self,G_not : np.ndarray = None, Energy : np.ndarray = None) -> np.ndarray:
        # call dyson
        norb = G_not.shape[0]
        nk = G_not.shape[3]
        nomega = G_not.shape[4]
        G_inv_not = np.zeros((norb,norb,self.ns,nk,nomega),dtype=complex,order='F')
        FLatTau_int = np.zeros((norb,norb,self.ns,nk,nomega),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,self.ns,nk,nomega),dtype=complex,order='F')
        
        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    G_inv_not[:,:,js,ik,iomega] = np.linalg.inv(G_not[:,:,js,ik,iomega])

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            tempmat[iorb,jorb,js,ik,iomega] = G_inv_not[iorb,jorb,js,ik,iomega]-Energy[iorb,jorb,js,ik,iomega]
        
        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    FLatTau_int[:,:,js,ik,iomega] = np.linalg.inv(tempmat[:,:,js,ik,iomega])
        
        return FLatTau_int    

    def Stc_self_energy(self,Sigma_C : np.ndarray)->np.ndarray:

        norb = Sigma_C.shape[0]
        ns = Sigma_C.shape[2]
        nk = Sigma_C.shape[3]
        nomega = Sigma_C.shape[4]

        Sigma_C_stc = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    tempmat[:,:,js,ik,iomega] = np.transpose(np.conjugate(Sigma_C[:,:,js,ik,iomega]))

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        Sigma_C_stc[iorb,jorb,js,ik] = (Sigma_C[iorb,jorb,js,ik,0]+tempmat[iorb,jorb,js,ik,0])/2
        
        return Sigma_C_stc
    
    def z_factor(self,Sigma : np.ndarray, beta : float)->np.ndarray:

        norb = Sigma.shape[0]
        ns = Sigma.shape[2]
        nk = Sigma.shape[3]
        nomega = Sigma.shape[4]

        Z = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        I = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    I[:,:,js,ik,iomega] = np.eye(norb,norb,dtype=complex,order="F")
                    tempmat[:,:,js,ik,iomega] = np.transpose(np.conjugate(Sigma[:,:,js,ik,iomega]))

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            tempmat2[iorb,jorb,js,ik] = (I[iorb,jorb,js,ik,iomega]-beta*(Sigma[iorb,jorb,js,ik,iomega]-tempmat[iorb,jorb,js,ik,iomega])/(2*np.pi))
        
        for ik in range(nk):
            for js in range(ns):
                Z[:,:,js,ik] = np.linalg.inv(tempmat2[:,:,js,ik])

        return Z
    
    def QP_Hamiltonian(self,H_not : np.ndarray, Hartree : np.ndarray, Fock : np.ndarray, Sigma_C : np.ndarray, mu : float, Z :np.ndarray)-> np.ndarray:

        norb = H_not.shape[0]
        ns = H_not.shape[2]
        nk = H_not.shape[3]

        H_QP = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for js in range(ns):
                eig_val, eig_vec = np.linalg.eig(Z[:,:,js,ik])
                for iorb in range(norb):
                    if 0<=(eig_val[iorb])<=1:
                        continue
                    else:
                        print("Error : The z-factor was calculated incorrectly. Please rerun the code.")
                        sys.exit()
                D = np.diag(eig_val)
                tempmat[:,:,js,ik] = np.dot(np.dot(eig_vec,np.sqrt(D)),np.linalg.inv(eig_vec))

        
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        tempmat2[iorb,jorb,js,ik] = H_not[iorb,jorb,js,ik] + Hartree[iorb,jorb,js,ik] + Fock[iorb,jorb,js,ik] + Sigma_C[iorb,jorb,js,ik] 
                        if iorb == jorb:
                            tempmat2[iorb,jorb,js,ik] = H_not[iorb,jorb,js,ik] + Hartree[iorb,jorb,js,ik] + Fock[iorb,jorb,js,ik] + Sigma_C[iorb,jorb,js,ik] -mu

        for ik in range(nk):
            for js in range(ns):
                H_QP[:,:,js,ik] = np.dot(np.dot(tempmat[:,:,js,ik],tempmat2[:,:,js,ik]),tempmat[:,:,js,ik])
        
        return H_QP
    
    def Energy_imp(self,Hmat : np.ndarray, mu : float,projector : np.ndarray):

        norb = Hmat.shape[0]
        ns = Hmat.shape[2]
        nk = Hmat.shape[3]
        nspace = projector.shape[3]
        norbc = projector.shape[1]

        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    Hmat[iorb,iorb,js,ik] -= mu
        
        E_imp = np.zeros((norbc,norbc,ns,nspace),dtype=complex,order='F')
        for ispace in range(nspace):
            E_imp[...,ispace] = DiagE.projection.flatstc(Hmat,projector[...,ispace])
        self.E_imp = E_imp
        return None
    
    def hybridisation(self,omega : np.ndarray, E_imp : np.ndarray,G_loc : np.ndarray, Sigma : np.ndarray):#, equiv : list):

        norb = G_loc.shape[0]
        ns = G_loc.shape[2]
        nfreq = G_loc.shape[3]
        

        beta = 1/(omega[0])*np.pi

        hyb = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')
        G_loc_inv = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')
        # Nind = np.amax(equiv)

        
        G_loc_inv = self.FInvLocDyn(G_loc)
        hyb_dic = {}
        
        for ifreq in range(nfreq):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        if iorb == jorb:
                            hyb[iorb,jorb,js,ifreq] = 1j*omega[ifreq] - E_imp[iorb,jorb,js] - G_loc_inv[iorb,jorb,js,ifreq] - Sigma[iorb,jorb,js,ifreq]
                        else:
                            hyb[iorb,jorb,js,ifreq] = - E_imp[iorb,jorb,js] - G_loc_inv[iorb,jorb,js,ifreq] - Sigma[iorb,jorb,js,ifreq]
        
        # for i in range(1,Nind+1):
        #     hyb_dic[str(i)]['beta'] = beta
        #     hyb_real = np.real(hyb).tolist()
        #     hyb_imag = np.imag(hyb).tolist()
        #     hyb_dic[str(i)]['imag'] = hyb_imag
        #     hyb_dic[str(i)]['real'] = hyb_real

        # f = open('hyb.json','w')
        # json.dump(hyb_dic,f,sort_keys='True',indent=4,separators=(', ',': '))
        # f.close()
        self.hyb = hyb
        return None
    
    def DC_self_energy(self,G_loc : np.ndarray, boson, IP)-> np.ndarray:

        norbc = G_loc.shape[0]
        ns = G_loc.shape[2]
        nf = G_loc.shape[3]
        norb = IP.bprojector.shape[1]
        nspace = IP.bprojector.shape[3]
        
        U = np.zeros((norb,norb,ns,ns,nspace),dtype=complex,order='F')
        # for iprob in range(nprob):
        #     U[:,:,:,:,iprob] = DiagE.projection.blatstc(boson.V_bare,IP.bprojector[:,:,:,iprob])
        for ispace in range(nspace):
            U[...,ispace] = DiagE.projection.blatstc(boson.V_bare,IP.bprojector[...,ispace])
        # U = DiagE.projection.blatstc(boson.V_bare,IP.bprojector)
        bf_list = boson.b2f
        norb = U.shape[0]
        
        
        Sigma1 = np.zeros((norbc,norbc,ns),dtype=complex,order='F')
        Sigma2 = np.zeros((norbc,norbc,ns),dtype=complex,order='F')
        Sigma = np.zeros((norbc,norbc,ns,nf,nspace),dtype=complex,order='F')

        ndim = norb*ns
        tempmat = np.zeros((ndim,ndim),dtype=complex,order='F')

        # for iprob in range(nprob):
        #     tempmat = boson.Bcomposite(U[:,:,:,:,iprob],1)
        for ispace in range(nspace):
            tempmat = boson.Bcomposite(U[...,ispace],1)

            nind = tempmat.shape[0]

            for ind1 in range(nind):
                nn1 = [0]*2
                ind1, [iorb,js] = indexing(nind,2,[norb,ns],0,ind1,nn1)
                iorbc1,iorbc2 = bf_list[iorb]
                for ind2 in range(nind):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = indexing(nind,2,[norb,ns],0,ind2,nn2)
                    iorbc3,iorbc4 = bf_list[jorb]
                    Sigma1[iorbc1,iorbc2,js] += tempmat[ind1,ind2]*G_loc[iorbc4,iorbc3,ks,-1,ispace]


            for js in range(ns):
                for iorb in range(norb):
                    iorbc1, iorbc4 = bf_list[iorb]
                    for jorb in range(norb):
                        iorbc3,iorbc2 = bf_list[jorb]
                        Sigma2[iorbc1,iorbc2,js] += G_loc[iorbc4,iorbc3,js,-1,ispace]*U[iorb,jorb,js,js,ispace]
            
            for iff in range(nf):
                Sigma[...,iff,ispace] = Sigma1 + Sigma2

        return Sigma

                        



class BHamiltonian(object):

    def __init__(self,crystal : Crystal, fermion :FHamiltonian = None):
        
        self.basis_f = crystal.basis_f
        self.basis_c = crystal.basis_c
        self.latt = crystal.latt
        self.bvec = crystal.bvec
        self.kpoint = crystal.kpoint
        self.kpath = crystal.kpath
        self.vol = crystal.vol
        self.rkgrid = crystal.grid
        self.ns = fermion.ns
        self.SOC = fermion.SOC
        self.fermion = fermion
        self.bind = {}
        self.orboption = {}
        self.int_term = []
        self.b2f = []
        self.c2b = []
        self.c2f = []
        self.V_loc = None
        self.V_nloc = None
        self.V_bare = None
    
    def set_basis_index(self,option : list):

        ind = []
        
        norb = len(self.bind)

        ii = 0
        orb_ind = list(range(option[1]))
        for m1, m2 in itertools.product(orb_ind,orb_ind):
            ind.append([option[0],[m1,m2]])

        for iorb in range(norb,norb+option[1]**2):
            self.bind[iorb] = ind[ii]
            ii+=1
            self.boson2fermion(iorb)
        self.composite2fermion()
        self.composite2boson()
        norb = len(self.bind)

        self.V_loc = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')

    def Batomorb(self,key : int) -> list:
        return self.bind[key]
    
    def Bindex(self,val:list) -> int:

        for key, value in self.bind.items():
            if val==value:
                return key
            
    def local_interacting(self,option : dict = None) -> np.ndarray:
        '''
        the orbital is fermionic orbital so have to translate to bosoninc orbital
        '''
        norbc = len(option["orbital"])
        ns = self.ns
        
        self.orboption['KorS'] = option["KorS"]
        self.orboption['params'] = {}


        if option["KorS"]=="K":
            self.orboption['params']['U'] = option["value"][0]
            self.orboption['params']['Up'] = option["value"][1]
            self.orboption['params']['J'] = option["value"][2]
            tempmat = self.Kanamori(norbc,option["value"])

            for iorbc in option["orbital"]:
                for jorbc in option["orbital"]:
                    for korbc in option["orbital"]:
                        for lorbc in option["orbital"]:
                            [a,m1] = self.fermion.Fatomorb(iorbc)
                            [b,m2] = self.fermion.Fatomorb(jorbc)
                            [b_prime,m3] = self.fermion.Fatomorb(korbc)
                            [a_prime,m4] = self.fermion.Fatomorb(lorbc)
                            if (a==a_prime)and(b==b_prime):
                                iorb = self.Bindex([a,[m1,m4]])
                                jorb = self.Bindex([b,[m2,m3]])
                                for s1, s2 in itertools.product(list(range(ns)),list(range(ns))):
                                    self.V_loc[iorb,jorb,s1,s2] = tempmat[m1,m2,m3,m4,s1,s2]
        elif option["KorS"]=="S":
            self.orboption['params']['F0'] = option["value"][0]
            self.orboption['params']['F2'] = option["value"][1]
            self.orboption['params']['F4'] = option["value"][2]
            tempmat = self.Slater_parameter(norbc,option["value"])
            for iorbc in option["orbital"]:
                for jorbc in option["orbital"]:
                    for korbc in option["orbital"]:
                        for lorbc in option["orbital"]:
                            [a,m1] = self.fermion.Fatomorb(iorbc)
                            [b,m2] = self.fermion.Fatomorb(jorbc)
                            [b_prime,m3] = self.fermion.Fatomorb(korbc)
                            [a_prime,m4] = self.fermion.Fatomorb(lorbc)
                            if (a==a_prime)and(b==b_prime):
                                iorb = self.Bindex([a,[m1,m4]])
                                jorb = self.Bindex([b,[m2,m3]])
                                for s1, s2 in itertools.product(list(range(ns)),list(range(ns))):
                                    self.V_loc[iorb,jorb,s1,s2] = tempmat[m1,m2,m3,m4,s1,s2]




    def Kanamori(self,norb : int = None, value : list = None) -> np.ndarray:
        print("Warning : In kanamori interaction, self interaction term has been added")

        V = np.zeros((norb,norb,norb,norb,self.ns,self.ns),dtype=float,order='F')
        U = value[0]
        U_prime = value[1]
        J = value[2]

        for m1 in range(norb):
            for m2 in range(norb):
                for m3 in range(norb):
                    for m4 in range(norb):
                        for s1 in range(self.ns):
                            for s2 in range(self.ns):
                                # if (m1==m2==m3==m4)and(s1==s2): # self interaction term
                                #     V[m1,m2,m3,m4,s1,s2] = U
                                if (m1==m2==m3==m4)and(s1!=s2):
                                    V[m1,m2,m3,m4,s1,s2] = U
                                elif (m1==m4)and(m2==m3)and(m1!=m2)and(s1!=s2):
                                    V[m1,m2,m3,m4,s1,s2] = U_prime # half or no
                                elif (m1==m4)and(m2==m3)and(m1!=m2)and(s1==s2):
                                    V[m1,m2,m3,m4,s1,s2] = (U_prime - J)
                                elif (m1==m3)and(m2==m4)and(m1!=m2)and(s1!=s2):
                                    V[m1,m2,m3,m4,s1,s2] = J
                                elif (m1==m2)and(m3==m4)and(m1!=m3)and(s1!=s2):
                                    V[m1,m2,m3,m4,s1,s2] = J


        return V*0.5

    def Slater_parameter(self,norb : int = None, radial_integral : list = None, SorC : str = "C"):

        V = np.zeros((norb,norb,norb,norb,self.ns,self.ns),dtype=float,order='F')

        l = int((norb-1)/2)
        m = list(range(-l,l+1))
        print(l,m)

        for n, F in enumerate(radial_integral):

            k = 2*n
            for m1 in m:
                for m2 in m:
                    for m3 in m:
                        for m4 in m:
                            for s1 in range(self.ns):
                                for s2 in range(self.ns):
                                    V[m1+l,m2+l,m3+l,m4+l,s1,s2] += F*self.angular_integral(l,k,m1,m2,m4,m3)
        if SorC == "C":
            for s1 in range(self.ns):
                for s2 in range(self.ns):
                    tempmat = V[:,:,:,:,s1,s2]
                    tempmat = self.tf_spherical_to_cubic(tempmat,l)
                    V[:,:,:,:,s1,s2] = tempmat
            return V
        else:
            return V

    def set_int_amp(self,int_amp : float = None,ind_i : int = None,ind_j : int = None,ind_R : list = None) -> list:
        
        new_int = [int_amp,int(ind_i),int(ind_j),np.array(ind_R)]

        self.int_term.append(new_int)

    def gen_nl_int_ham(self) -> np.ndarray:

        kgrid = self.rkgrid
        kgrid = np.array(kgrid)
        nk = kgrid[0]*kgrid[1]*kgrid[2]

        kvec = np.array(list(itertools.product(np.arange(0,kgrid[2])/kgrid[2],np.arange(0,kgrid[1])/kgrid[1],np.arange(0,kgrid[0])/kgrid[0])))
        kvec = np.fliplr(kvec)

        norb = len(self.bind)

        self.V_nloc = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

        for int_term in self.int_term:
            amp = int_term[0]

            iorb = int_term[1]
            jorb = int_term[2]

            [alpha,[m1,m4]] = self.Batomorb(iorb)
            [beta,[m2,m3]] = self.Batomorb(jorb)

            R = np.array(int_term[3])

            rv = self.basis_f[alpha,:]-self.basis_f[beta,:]+R
            phase = np.exp(-2.0j*np.pi*np.dot(kvec,rv))

            for ik in range(nk):
                for s1 in range(self.ns):
                    for s2 in range(self.ns):
                        self.V_nloc[iorb,jorb,s1,s2,ik] += amp*phase[ik]
                        self.V_nloc[jorb,iorb,s1,s2,ik] += amp*phase[ik].conjugate()

        

    def Combine_interaction(self) -> np.ndarray:
        '''
        combine in k-space
        '''

        nk = self.V_nloc.shape[4]
        norb = self.V_loc.shape[0]
        self.V_bare = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')
        for ik in range(nk):
            self.V_bare[:,:,:,:,ik] = self.V_loc+self.V_nloc[:,:,:,:,ik]
        
        


    def angular_integral(self,l,k,m1,m2,m3,m4):
        ang_int = 0
        pi = np.pi

        for q in range(-k,k+1):
            ang_int += gaunt(l,k,l,-m1,q,m3)*np.conjugate(gaunt(l,k,l,m4,-q,-m2))*((-1.0 if(m1+q+m2)%2 == 1 else 1.0))

        ang_int *= 4*pi/(2*k+1)

        return ang_int

    def rotaion_matrix(self,l):

        m_range = 2*l+1

        R = np.zeros((m_range,m_range),dtype=complex)
        
        if l == 0:
           R = np.eye(m_range,m_range,dtype=complex)

        elif l == 1:
            '''/n
            py, pz, px
            '''
            R[0,0] = 1j/np.sqrt(2)
            R[2,0] = 1j/np.sqrt(2)

            R[1,1] = 1

            R[0,2] = 1/np.sqrt(2)
            R[2,2] = -1/np.sqrt(2)

        elif l==2:
            '''/n
            xy, yz, z^2, xz, x^2-y^2
            '''

            R[0,0] = 1j/np.sqrt(2)
            R[4,0] = -1j/np.sqrt(2)

            R[1,1] = 1j/np.sqrt(2)
            R[3,1] = 1j/np.sqrt(2)

            R[2,2] = 1

            R[1,3] = 1/np.sqrt(2)
            R[3,3] = -1/np.sqrt(2)

            R[0,4] = 1/np.sqrt(2)
            R[4,4] = 1/np.sqrt(2)

        elif l==3:
            '''/n
            3x^2-y^2 xyz yz^2 xz^2 z(x^2-y^2) x(x^2-3y^2)
            '''

            R[0,0] = 1j/np.sqrt(2)
            R[6,0] = 1j/np.sqrt(2)

            R[1,1] = 1j/np.sqrt(2)
            R[5,1] = -1j/np.sqrt(2)

            R[2,2] = 1j/np.sqrt(2)
            R[4,2] = 1j/np.sqrt(2)

            R[3,3] = 1

            R[2,4] = 1/np.sqrt(2)
            R[4,4] = -1/np.sqrt(2)

            R[1,5] = 1/np.sqrt(2)
            R[5,5] = 1/np.sqrt(2)

            R[0,6] = 1/np.sqrt(2)
            R[6,6] = -1/np.sqrt(2)

        return R

    def tf_spherical_to_cubic(self,V=None,l=None):

        R = self.rotaion_matrix(l)

        R_dag = np.conjugate(np.transpose(R))

        tempmat = np.einsum("ab,cd,bdeg,ef,gh",R_dag,R_dag,V,R,R)
        tempmat = np.real(tempmat)

        V = np.array(tempmat,dtype=float,order='F')

        return V
    
    def get_Uijkl_comctqmc(self,key):
        
        
        V_loc_temp = self.Convert_4_2_LocStc(self.V_loc,0)
        orb = self.imp_dict[key][0]
        norb = len(orb)
        ns = self.ns
        V_loc = np.zeros((norb,norb,norb,norb,ns,ns),dtype=complex)
        for js in range(ns):
            for ks in range(ns):
                for ii,iorb in enumerate(orb):
                    for jj, jorb in enumerate(orb):
                        for kk, korb in enumerate(orb):
                            for ll, lorb in enumerate(orb):
                                V_loc[ii,jj,kk,ll,js,ks] = V_loc_temp[iorb,jorb,korb,lorb,js,ks]
        nimp_orb = V_loc.shape[0]
        
        if self.fermion.SOC is False:
            U = np.zeros(nimp_orb**4*2**4,dtype=float)
            index = 0
            if self.ns == 1:
                for sl in range(2):
                    for l in range(nimp_orb):
                        for sk in range(2):
                            for k in range(nimp_orb):
                                for sj in range(2):
                                    for j in range(nimp_orb):
                                        for si in range(2):
                                            for i in range(nimp_orb):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val=V_loc[i,j,k,l,0,0].real
                                                    val=abs(val)
                                                    if(val > 0.001):
                                                        U[index]=val
                                                index=index+1
            elif self.ns == 2:
                for sl in range(2):
                    for l in range(nimp_orb):
                        for sk in range(2):
                            for k in range(nimp_orb):
                                for sj in range(2):
                                    for j in range(nimp_orb):
                                        for si in range(2):
                                            for i in range(nimp_orb):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val=V_loc[i,j,k,l,si,sj].real
                                                    val=abs(val)
                                                    if(val > 0.001):
                                                        U[index]=val
                                                index=index+1
        else:
            print("SOC is not False")
            sys.exit()
        self.U_ctqmc = U
        
        return None
    
    def boson2fermion(self,ind : int):

        [a, [m1,m2]] = self.Batomorb(ind)
        iorbc1 = self.fermion.Findex([a,m1])
        iorbc2 = self.fermion.Findex([a,m2])
        self.b2f.append([iorbc1,iorbc2])
    
    def composite2fermion(self):

        norbc = len(self.fermion.find)
        norb = norbc*norbc
        c2f = []

        for iorbc in range(norbc):
            for jorbc in range(norbc):
                nn1 = [iorbc,jorbc]
                iorb, nn1 = indexing(norb,2,[norbc,norbc],1,0,nn1)
                c2f.append([iorbc,jorbc])
        self.c2f = c2f
    
    def composite2boson(self):

        norbc = len(self.fermion.find)
        ndim = norbc*norbc
        c2b = []

        for ind in range(ndim):
            nn1 = [0]*2
            ind,[iorbc,jorbc] = indexing(ndim,2,[norbc,norbc],0,ind,nn1)
            [a,m1] = self.fermion.Fatomorb(iorbc)
            [a_p,m2] = self.fermion.Fatomorb(jorbc)
            if a==a_p:
                borb = self.Bindex([a,[m1,m2]])
                if borb is not None:
                    c2b.append([borb,ind])
        self.c2b = c2b

    def Bcomposite(self, mat : np.ndarray, flag : int):

        if flag ==1 :
            norb = mat.shape[0]
            ns = mat.shape[2]
            mat_out = np.zeros((norb*ns,norb*ns),dtype=complex,order='F')
            
            for js in range(ns):
                for iorb in range(norb):
                    nn1 = [iorb,js]
                    ind1, nn1 = indexing(norb*ns,2,[norb,ns],flag,0,nn1)
                    for ks in range(ns):
                        for jorb in range(norb):
                            nn2 = [jorb,ks]
                            ind2, nn2 = indexing(norb*ns,2,[norb,ns],flag,0,nn2)
                            mat_out[ind1,ind2] = mat[iorb,jorb,js,ks]
            return mat_out
        elif flag == 0:
            norb = len(self.bind)
            ns = self.ns
            mat_out = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')
            ndim = mat.shape[0]

            for ind1 in range(ndim):
                nn1 = [0]*2
                ind1, [iorb,js] = indexing(ndim,2,[norb,ns],flag,ind1,nn1)
                for ind2 in range(ndim):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = indexing(ndim,2,[norb,ns],flag,ind2,nn2)
                    mat_out[iorb,jorb,js,ks] = mat[ind1,ind2]

            return mat_out

    
    def Convert_4_2(self,mat : np.ndarray = None, flag : int = None) -> np.ndarray: # 4 index <-> 2 index

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        if flag == 1:
            

            mat_ret = np.zeros((norb,norb),dtype=complex)

            for iorb, [iorbc,lorbc] in enumerate(self.b2f):
                for jorb, [jorbc,korbc] in enumerate(self.b2f):
                    mat_ret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

            return mat_ret
        
        elif flag == 0:
            


            mat_ret = np.zeros((norbc,norbc,norbc,norbc),dtype=complex,order='F')

            for iorb, [iorbc,lorbc] in enumerate(self.b2f):
                for jorb, [jorbc,korbc] in enumerate(self.b2f):
                    mat_ret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]

            return mat_ret
        
    def Convert_4_2_LocStc(self,mat : np.ndarray = None, flag : int = None) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)

        if flag == 1:

            mat_ret = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')

            for js in range(self.ns):
                for ks in range(self.ns):
                    mat_ret[:,:,js,ks] = self.Convert_4_2(mat[:,:,:,:,js,ks],flag)

            return mat_ret
        elif flag == 0:

            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns),dtype=complex,order='F')

            for js in range(self.ns):
                for ks in range(self.ns):
                    mat_ret[:,:,:,:,js,ks] = self.Convert_4_2(mat[:,:,js,ks],flag)

            return mat_ret
        
    def Convert_4_2_LatStc(self,mat : np.ndarray, flag :int) -> np.ndarray:
        
        norb = len(self.bind)
        norbc = len(self.fermion.find)

        if flag == 1:
            nk = mat.shape[6]
            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')
            for ik in range(nk):
                mat_ret[:,:,:,:,ik] = self.Convert_4_2_LocStc(mat[:,:,:,:,:,:,ik],flag)
            
            return mat_ret
        
        elif flag == 0:
            nk = mat.shape[4]
            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nk),dtype=complex,order='F')

            for ik in range(nk):
                mat_ret[:,:,:,:,:,:,ik] = self.Convert_4_2_LocStc(mat[:,:,:,:,ik],flag)
            
            return mat_ret
        
    def Convert_4_2_LocDyn(self,mat : np.ndarray, flag :int) -> np.ndarray:
        
        norb = len(self.bind)
        norbc = len(self.fermion.find)

        if flag == 1:
            nt = mat.shape[6]
            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nt),dtype=complex,order='F')
            for it in range(nt):
                mat_ret[:,:,:,:,it] = self.Convert_4_2_LocStc(mat[:,:,:,:,:,:,it],flag)
            
            return mat_ret
        
        elif flag == 0:
            nt = mat.shape[4]
            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,:,it] = self.Convert_4_2_LocStc(mat[:,:,:,:,it],flag)
            
            return mat_ret
    
    def Convert_4_2_LatDyn(self,mat : np.ndarray, flag :int) -> np.ndarray:
        
        norb = len(self.bind)
        norbc = len(self.fermion.find)
    
        if flag == 1:

            nk = mat.shape[6]
            nt = mat.shape[7]
            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nk,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,it] = self.Convert_4_2_LatStc(mat[:,:,:,:,:,:,:,it],flag)
            
            return mat_ret
        elif flag == 0:

            nk = mat.shape[4]
            nt = mat.shape[5]
            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nk,nt),dtype=complex,order='F')
            
            for it in range(nt):
                mat_ret[:,:,:,:,:,:,:,it] = self.Convert_4_2_LatStc(mat[:,:,:,:,:,it],flag)

            return mat_ret
    
    def full_2_4(self,mat : np.ndarray, flag : int) -> np.ndarray: # 4index <-> new 2 index

        norbc = len(self.fermion.find)
        norb = norbc*norbc
        

        if flag == 1:
            mat_ret = np.zeros((norb,norb),dtype=complex,order='F')
            for iorb,[iorbc,lorbc] in enumerate(self.c2f):
                for jorb, [jorbc,korbc] in enumerate(self.c2f):
                    mat_ret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

            return mat_ret
        elif flag == 0:
            mat_ret = np.zeros((norbc,norbc,norbc,norbc),dtype=complex,order='F')
            for iorb, [iorbc,lorbc] in enumerate(self.c2f):
                for jorb, [jorbc,korbc] in enumerate(self.c2f):
                    mat_ret[iorbc,jorbc,korbc,lorbc] = mat_ret[iorb,jorb]
            
            return mat_ret
    
    def full_2_4LocStc(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norbc = len(self.fermion.find)
        norb = norbc*norbc

        if flag == 1:
            
            
            mat_ret = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
            
            for js in range(self.ns):
                for ks in range(self.ns):
                    mat_ret[:,:,js,ks] = self.full_2_4(mat[:,:,:,:,js,ks],flag)
            
            return mat_ret
        
        elif flag == 0:

            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns),dtype=complex,order='F')

            for js in range(self.ns):
                for ks in range(self.ns):
                    mat_ret[:,:,:,:,js,ks] = self.full_2_4(mat[:,:,js,ks],flag)

            return mat_ret
    
    def full_2_4LatStc(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norbc = len(self.fermion.find)
        norb = norbc*norbc

        if flag == 1:

            nk = mat.shape[6]

            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

            for ik in range(nk):
                mat_ret[:,:,:,:,ik] = self.full_2_4LocStc(mat[:,:,:,:,:,:,ik],flag)
            
            return mat_ret
        elif flag == 0:

            nk = mat.shape[4]

            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nk),dtype=complex,order='F')

            for ik in range(nk):
                mat_ret[:,:,:,:,:,:,ik] = self.full_2_4LocStc(mat[:,:,:,:,ik],flag)

            return mat_ret
    
    def full_2_4LocDyn(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norbc = len(self.fermion.find)
        norb = norbc*norbc

        if flag == 1:

            nt = mat.shape[6]

            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,it] = self.full_2_4LocStc(mat[:,:,:,:,:,:,it],flag)
            
            return mat_ret
        elif flag == 0:

            nt = mat.shape[4]

            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,:,it] = self.full_2_4LocStc(mat[:,:,:,:,it],flag)

            return mat_ret
        
    def full_2_4LatDyn(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norbc = len(self.fermion.find)
        norb = norbc*norbc

        if flag == 1:

            nk = mat.shape[6]
            nt = mat.shape[7]

            mat_ret = np.zeros((norb,norb,self.ns,self.ns,nk,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,it] = self.full_2_4LatStc(mat[:,:,:,:,:,:,:,it],flag)

            return mat_ret
        elif flag == 0:

            nk = mat.shape[4]
            nt = mat.shape[5]

            mat_ret = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nk,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,:,:,it] = self.full_2_4LatStc(mat[:,:,:,:,:,it],flag)

            return mat_ret
    
    def full_2_2(self, mat : np.ndarray, flag : int) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        nind = norbc*norbc
        

        if flag == 1:
            mat_ret = np.zeros((norb,norb),dtype=complex,order='F')
            for iorb, ind1 in self.c2b:
                for jorb, ind2 in self.c2b:
                    mat_ret[iorb,jorb] = mat[ind1, ind2]
            
            return mat_ret
        elif flag == 0:
            mat_ret = np.zeros((nind,nind),dtype=complex,order='F')
            for iorb, ind1 in enumerate(self.c2b):
                for jorb, ind2 in enumerate(self.c2b):
                    mat_ret[ind1,ind2] = mat[iorb,jorb]

            return mat_ret
    
    def full_2_2LocStc(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        nind = norbc*norbc
        ns = self.ns

        if flag == 1:
            
            
            mat_ret = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')

            for js in range(ns):
                for ks in range(ns):
                    mat_ret[:,:,js,ks] = self.full_2_2(mat[:,:,js,ks],flag)

            return mat_ret
        elif flag == 0:

            mat_ret = np.zeros((nind,nind,ns,ns),dtype=complex,order='F')

            for js in range(ns):
                for ks in range(ns):
                    mat_ret[:,:,js,ks] = self.full_2_2(mat[:,:,js,ks],flag)

            return mat_ret
    
    def full_2_2LatStc(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        nind = norbc*norbc
        ns = self.ns
        nk = mat.shape[4]

        if flag == 1:
            
            mat_ret = np.zeros((norb,norb,ns,ns,nk),dtype=complex,order='F')
            for ik in range(nk):
                mat_ret[:,:,:,:,ik] = self.full_2_2LocStc(mat[:,:,:,:,ik], flag)
            
            return mat_ret
        elif flag == 0:

            mat_ret = np.zeros((nind,nind,ns,ns,nk),dtype=complex,order='F')
            for ik in range(nk):
                mat_ret[:,:,:,:,ik] = self.full_2_2LocStc(mat[:,:,:,:,ik], flag)
            
            return mat_ret
    

    def full_2_2LocDyn(self,mat : np.ndarray, flag : int) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        nind = norbc*norbc
        ns = self.ns
        nt = mat.shape[4]

        if flag == 1:
            
            mat_ret = np.zeros((norb,norb,ns,ns,nt),dtype=complex,order='F')
            for it in range(nt):
                mat_ret[:,:,:,:,it] = self.full_2_2LocStc(mat[:,:,:,:,it], flag)
            
            return mat_ret
        elif flag == 0:

            mat_ret = np.zeros((nind,nind,ns,ns,nt),dtype=complex,order='F')
            for it in range(nt):
                mat_ret[:,:,:,:,it] = self.full_2_2LocStc(mat[:,:,:,:,it], flag)
            
            return mat_ret
    
    def full_2_2LatDyn(self, mat : np.ndarray, flag : int) -> np.ndarray:

        norb = len(self.bind)
        norbc = len(self.fermion.find)
        nind = norbc*norbc
        ns = self.ns
        nk = mat.shape[4]
        nt = mat.shape[5]

        if flag == 1:

            mat_ret = np.zeros((norb,norb,ns,ns,nk,nt),dtype=complex, order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,it] = self.full_2_2LatStc(mat[:,:,:,:,:,it],flag)
            
            return mat_ret
        
        elif flag == 0:

            mat_ret = np.zeros((nind,nind,ns,ns,nk,nt),dtype=complex,order='F')

            for it in range(nt):
                mat_ret[:,:,:,:,:,it] = self.full_2_2LatStc(mat[:,:,:,:,:,it],flag)

            return mat_ret
    
    def BimpconvertLocStc(self,mat : np.ndarray, flag : int):

        norb = mat.shape[0]
        ns = mat.shape[2]
        
        if flag == 1:#problem -> space

            nspace = 0
            for key, val in self.prob_space.items():
                nspace += len(val)
            mat_ret = np.zeros((norb,norb,ns,ns,nspace),dtype=complex,order='F')
            
            for key,val in self.prob_space.items():
                iprob = int(key)-1
                for ispace in val:
                    mat_ret[...,ispace] = mat[...,iprob]
            return mat_ret
        
        elif flag==0: # space -> problem
            
            nprob = len(self.prob_space)
            mat_ret = np.zeros((norb,norb,ns,ns,nprob),dtype=complex,order='F')
            
            for key,val in self.prob_space.items():
                iprob = int(key)-1
                tempmat = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')
                for ispace in val:
                    tempmat += mat[...,ispace]
                tempmat /= len(val)

                mat_ret[...,iprob] = tempmat
            return mat_ret
            
        
    def BimpconvertLocDyn(self,mat : np.ndarray, flag : int):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[4]


        if flag == 1:
            nspace = 0
            for key, val in self.prob_space.items():
                nspace += len(val)
            mat_ret = np.zeros((norb,norb,ns,ns,nft,nspace),dtype=complex,order='F')
            for ift in range(nft):
                mat_ret[:,:,:,:,ift,:] = self.BimpconvertLocStc(mat[:,:,:,:,ift,:],flag)
            return mat_ret
        if flag == 0:
            nprob = len(self.prob_space)
            mat_ret = np.zeros((norb,norb,ns,ns,nft,nprob),dtype=complex,order='F')
            for ift in range(nft):
                mat_ret[:,:,:,:,ift,:] = self.BimpconvertLocStc(mat[:,:,:,:,ift,:],flag)
            return mat_ret




    def BMulLocStc(self, mat1 : np.ndarray, mat2 : np.ndarray)-> np.ndarray:

        norb = mat1.shape[0]
        ns = self.ns

        mat_out = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')

        ndim = norb*ns
        tempmat = np.zeros((ndim,ndim),dtype=complex)
        tempmat2 = np.zeros((ndim,ndim),dtype=complex)
        tempmat3 = np.zeros((ndim,ndim),dtype=complex)

        for js in range(ns):
            for iorb in range(norb):
                nn1 = [iorb,js]
                ind1, nn1 = indexing(ndim,2,[norb,ns],1,0,nn1)
                for ks in range(ns):
                    for jorb in range(norb):
                        nn2 = [jorb,ks]
                        ind2, nn2 = indexing(ndim,2,[norb,ns],1,0,nn2)
                        tempmat[ind1,ind2] = mat1[iorb,jorb,js,ks]
                        tempmat2[ind1,ind2] = mat2[iorb,jorb,js,ks]

        tempmat3 = tempmat@tempmat2

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb,js] = indexing(ndim,2,[norb,ns],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb,ks] = indexing(ndim,2,[norb,ns],0,ind2,nn2)
                mat_out[iorb,jorb,js,ks] = tempmat3[ind1,ind2]
        
        return mat_out
    
    def BMulLatStc(self,mat1 : np.ndarray, mat2 : np.ndarray)-> np.ndarray:
        
        norb = mat1.shape[0]
        nk = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            mat_out[:,:,:,:,ik] = self.BMulLocStc(mat1[:,:,:,:,ik],mat2[:,:,:,:,ik])

        return mat_out
    
    def BMulLocDyn(self,mat1 : np.ndarray, mat2 : np.ndarray)-> np.ndarray:

        norb = mat1.shape[0]
        nt = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nt),dtype=complex, order='F')

        for it in range(nt):
            mat_out[:,:,:,:,it] = self.BMulLocStc(mat1[:,:,:,:,it],mat2[:,:,:,:,it])

        return mat_out
    
    def BMulLatDyn(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nk = mat1.shape[4]
        nt = mat1.shape[5]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_out[:,:,:,:,:,it] = self.BMulLatStc(mat1[:,:,:,:,:,it],mat2[:,:,:,:,:,it])
        
        return mat_out
    
    def BImMLocStc(self,mat1 : np.ndarray, mat2 : np.ndarray)-> np.ndarray:

        norb = mat1.shape[0] # new 2 index
        ns = self.ns
        mat_out = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')
        I = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')
        tempmat = np.eye(norb*ns,norb*ns,dtype=complex)
        ndim = norb*ns

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb,js] = indexing(ndim,2,[norb,ns],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb,ks] = indexing(ndim,2,[norb,ns],0,ind2,nn2)
                I[iorb,jorb,js,ks] = tempmat[ind1,ind2]
        
        tempmat2 = self.BMulLocStc(mat1,mat2)

        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        mat_out[iorb,jorb,js,ks] = I[iorb,jorb,js,ks] - tempmat2[iorb,jorb,js,ks]

        return mat_out
    
    def BImMLatStc(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nk = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            mat_out[:,:,:,:,ik] = self.BImMLocStc(mat1[:,:,:,:,ik],mat2[:,:,:,:,ik])

        return mat_out
    
    def BImMLocDyn(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nt = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_out[:,:,:,:,it] = self.BImMLocStc(mat1[:,:,:,:,it],mat2[:,:,:,:,it])

        return mat_out
    
    def BImMLatDyn(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nk = mat1.shape[4]
        nt = mat1.shape[5]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_out[:,:,:,:,:,it] = self.BImMLatStc(mat1[:,:,:,:,:,it],mat2[:,:,:,:,:,it])

        return mat_out
    
    def BIimMLocStc(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        ns = self.ns

        mat_out = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=complex,order='F')
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=complex,order='F')
        

        mat_temp = self.BImMLocStc(mat1,mat2)
        ndim = norb*ns

        for js in range(ns):
            for iorb in range(norb):
                nn1 = [iorb,js]
                ind1, nn1 = indexing(ndim,2,[norb,ns],1,0,nn1)
                for ks in range(ns):
                    for jorb in range(norb):
                        nn2 = [jorb,ks]
                        ind2, nn2 = indexing(ndim,2,[norb,ns],1,0,nn2)
                        tempmat[ind1,ind2] = mat_temp[iorb,jorb,js,ks]
        
        tempmat2 = np.linalg.inv(tempmat)

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb,js] = indexing(ndim,2,[norb,ns],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb,ks] = indexing(ndim,2,[norb,ns],0,ind2,nn2)
                mat_out[iorb,jorb,js,ks] = tempmat2[ind1,ind2]

        return mat_out
    
    def BIimMLatStc(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nk = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            mat_out[:,:,:,:,ik] = self.BIimMLocStc(mat1[:,:,:,:,ik],mat2[:,:,:,:,ik])

        return mat_out
    
    def BIimMLocDyn(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nt = mat1.shape[4]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_out[:,:,:,:,it] = self.BIimMLocStc(mat1[:,:,:,:,it],mat2[:,:,:,:,it])

        return mat_out
    
    def BIimMLatDyn(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]
        nk = mat1.shape[4]
        nt = mat1.shape[5]

        mat_out = np.zeros((norb,norb,self.ns,self.ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            mat_out[:,:,:,:,:,it] = self.BIimMLatStc(mat1[:,:,:,:,:,it],mat2[:,:,:,:,:,it])

        return mat_out

    def BLatStc_K2R(self,rkgrid : list = None, hmatk : np.ndarray = None) -> np.ndarray:

        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nrk = rkgrid[0]*rkgrid[1]*rkgrid[2]
        norb = hmatk.shape[0]


        rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
        rkvec = np.fliplr(rkvec)

        if (norb==len(self.bind)):
            for iorb in range(norb):
                for jorb in range(norb):

                    alpha,[m1,m2] = self.Batomorb(iorb)
                    beta,[m3,m4] = self.Batomorb(jorb)

                    delta = self.basis_f[alpha,:] - self.basis_f[beta,:]

                    phase = np.exp(2.0j*np.pi*np.dot(rkvec,delta))

                    for ik in range(nrk):
                        hmatk[iorb,jorb,:,:,ik] *= phase[ik]
            hmatr = DiagE.fourier.blatstc_k2r(rkgrid,hmatk)
        else:
            tempmat = hmatk
            hmatk = np.zeros((len(self.bind),len(self.bind),self.ns,self.ns,nrk),dtype=complex,order='F')
            for js in range(self.ns):
                for ks in range(self.ns):
                    for iorb,ind1 in enumerate(self.c2b):
                        for jorb,ind2 in enumerate(self.c2b):
                            
                            alpha,[m1,m4] = self.Batomorb(iorb)
                            beta,[m2,m3] = self.Batomorb(jorb)

                            delta = self.basis_f[alpha,:] - self.basis_f[beta,:]

                            phase = np.exp(2.0j*np.pi*np.dot(rkvec,delta))
                            
                            for ik in range(nrk):
                                hmatk[iorb,jorb,js,ks,ik] = tempmat[ind1,ind2,js,ks,ik]*phase[ik]
            
            hmatr = DiagE.fourier.blatstc_k2r(rkgrid,hmatk)

            tempmat = hmatr
            hmatr = np.zeros((norb,norb,self.ns,self.ns,nrk),dtype=complex,order='F')

            for ir in range(nrk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb, ind1 in enumerate(self.c2b):
                            for jorb, ind2 in enumerate(self.c2b):
                                hmatr[ind1,ind2,js,ks,ir] = tempmat[iorb,jorb,js,ks,ir]

        

        return hmatr

    def BLatStc_R2K(self,rkgrid : list = None, hmatr : np.ndarray = None) -> np.ndarray:

        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nrk = rkgrid[0]*rkgrid[1]*rkgrid[2]
        norb = hmatr.shape[0]

        hmatk = DiagE.fourier.blatstc_r2k(rkgrid,hmatr)

        rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
        rkvec = np.fliplr(rkvec)


        if (norb==len(self.bind)):
            for iorb in range(norb):
                for jorb in range(norb):

                    alpha,[m1,m2] = self.Batomorb(iorb)
                    beta,[m3,m4] = self.Batomorb(jorb)

                    delta = self.basis_f[alpha,:]-self.basis_f[beta,:]

                    phase = np.exp(-2.0j*np.pi*np.dot(rkvec,delta))

                    for ir in range(nrk):
                        hmatk[iorb,jorb,:,:,ir] *= phase[ir]
        else:
            tempmat = hmatk
            hmatk = np.zeros((len(self.bind),len(self.bind),self.ns,self.ns,nrk),dtype=complex,order='F')

            for js in range(self.ns):
                for ks in range(self.ns):
                    for iorb, ind1 in enumerate(self.c2b):
                        for jorb, ind2 in enumerate(self.c2b):

                            alpha,[m1,m4] = self.Batomorb(iorb)
                            beta,[m2,m3] = self.Batomorb(jorb)

                            delta = self.basis_f[alpha,:] - self.basis_f[beta,:]

                            phase = np.exp(-2.0j*np.pi*np.dot(rkvec,delta))

                            for ik in range(nrk):
                                hmatk[iorb,jorb,js,ks,ik] = tempmat[ind1,ind2,js,ks,ik]*phase[ik]
            
            tempmat = hmatk
            hmatk = np.zeros((norb,norb,self.ns,self.ns,nrk),dtype=complex,order='F')

            for ik in range(nrk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb, ind1 in enumerate(self.c2b):
                            for jorb, ind2 in enumerate(self.c2b):
                                hmatk[ind1,ind2,js,ks,ik] = tempmat[iorb,jorb,js,ks,ik]
            

        return hmatk

    def BLatDyn_K2R(self,rkgrid : list = None, hmatk : np.ndarray = None) -> np.ndarray:
        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nto = hmatk.shape[5]
        norb = hmatk.shape[0]
        ns = hmatk.shape[2]
        nk = hmatk.shape[4]
        hmatr = np.zeros((norb,norb,ns,ns,nk,nto),dtype=complex,order='F')

        for ito in range(nto):
            hmatr[:,:,:,:,:,ito] = self.BLatStc_K2R(rkgrid,hmatk[:,:,:,:,:,ito])

        return hmatr

    def BLatDyn_R2K(self,rkgrid : list = None, hmatr : np.ndarray = None) -> np.ndarray:
        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nto = hmatr.shape[5]
        norb = hmatr.shape[0]
        ns = hmatr.shape[2]
        nk = hmatr.shape[4]
        hmatk = np.zeros((norb,norb,ns,ns,nk,nto),dtype=complex,order='F')

        for ito in range(nto):
            hmatk[:,:,:,:,:,ito] = self.BLatStc_R2K(rkgrid,hmatr[:,:,:,:,:,ito])

        return hmatk
    
    def BLocDyn_M(self, nu : np.ndarray = None, ff : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:
        
        norb = ff.shape[0]
        

        momentum = np.zeros((norb,norb,self.ns,self.ns,3),dtype=complex,order='F')
        high = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
        momentum, high = DiagE.fourier.blocdyn_m(nu,ff,oddzero,highzero)

        return momentum,high
    
    def BLatDyn_M(self, nu : np.ndarray = None, ff : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:
        
        norb = ff.shape[0]
        nk = ff.shape[4]
        

        momentum = np.zeros((norb,norb,self.ns,self.ns,nk,3),dtype=complex,order='F')
        high = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')
        momentum, high = DiagE.fourier.blatdyn_m(nu,ff,oddzero,highzero)

        return momentum,high
    
    def BLocDyn_T2F(self,tau : np.ndarray = None, btau : np.ndarray = None, freq : np.ndarray = None) -> np.ndarray:

        bf = np.empty_like(btau,dtype=complex,order='F')

        bf = DiagE.fourier.blocdyn_t2f(tau,btau,freq)

        return bf
    
    def BLatDyn_T2F(self,tau : np.ndarray = None, btau : np.ndarray = None, freq : np.ndarray = None) -> np.ndarray:

        nk = btau.shape[4]
        
        bf  = np.empty_like(btau,dtype=complex,order='F')

        bf = DiagE.fourier.blatdyn_t2f(tau,btau,freq)

        return bf
    
    def BLocDyn_F2T(self,freq : np.ndarray = None,bnu : np.ndarray = None, tau : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:

        momentum, high = self.BLocDyn_M(freq,bnu,oddzero,highzero)

        btau = np.empty_like(bnu, dtype=complex,order='F')
        btau = DiagE.fourier.blocdyn_f2t(freq,bnu,momentum,tau)

        return btau
    
    def BLatDyn_F2T(self,freq : np.ndarray = None,bnu : np.ndarray = None, tau : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:

        nk = bnu.shape[4]
        momentum, high = self.BLatDyn_M(freq,bnu,oddzero,highzero)
        btau = np.empty_like(bnu,dtype=complex,order='F')

        btau = DiagE.fourier.blatdyn_f2t(freq,bnu,momentum,tau)
        
        return btau
    
    def BmixLocStc(self,iter : int , mix : float, Bb : np.ndarray, Bm : np.ndarray)->np.ndarray:
        norb = Bb.shape[0]
        ns = Bb.shape[2]

        B_new = np.zeros((norb,norb,ns,ns),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0
        
        for js in range(ns):
            for ks in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        B_new[iorb,jorb,js,ks] = mix*Bb[iorb,jorb,js,ks]+(1-mix)*Bm[iorb,jorb,js,ks]

        return B_new

    def BmixLatStc(self, iter : int, mix : float, Bb : np.ndarray, Bm : np.ndarray)->np.ndarray:
        
        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]

        B_new = np.zeros((norb,norb,ns,ns,nrk),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0
        
        

        for irk in range(nrk):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            B_new[iorb,jorb,js,ks,irk] = mix*Bb[iorb,jorb,js,ks,irk]+(1-mix)*Bm[iorb,jorb,js,ks,irk]
        
        return B_new
    
    def BmixLocDyn(self, iter : int, mix :float, Bb : np.ndarray, Bm : np.ndarray)->np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nft = Bb.shape[4]

        B_new = np.zeros((norb,norb,ns,ns,nft),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0

        for ift in range(nft):
            for js in range(ns):
                for ks in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            B_new[iorb,jorb,js,ks,ift] = mix*Bb[iorb,jorb,js,ks,ift] + (1-mix)*Bm[iorb,jorb,js,ks,ift]
        
        return B_new
    
    def BmixLatDyn(self, iter : int, mix : float, Bb : np.ndarray, Bm : np.ndarray)->np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nrk = Bb.shape[4]
        nft = Bb.shape[5]

        B_new = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=complex,order='F')

        if iter == 1:
            mix = 1.0

        for ift in range(nft):
            for irk in range(nrk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                B_new[iorb,jorb,js,ks,irk,ift] = mix*Bb[iorb,jorb,js,ks,irk,ift]+(1-mix)*Bm[iorb,jorb,js,ks,irk,ift]
        
        return B_new

    def BgaussianLocDyn(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]
        ynew = np.zeros((norb,norb,ns,ns,nft),dtype=complex,order='F')
        w0 = (1.0-3.0*w1)*np.pi*temperature
        width_array = w0+w1*x
        cnt = 0
        for x0 in x:
            if (x0>cutoff+(w0+w1*cutoff)*3.0):
                ynew[...,cnt] = y[...,cnt]
            else:
                if ((x0>3*width_array[cnt])and((x[-1]-x0)>3*width_array[cnt])):
                    dist = 1.0/np.sqrt(2*np.pi)/width_array[cnt]*np.exp(-(x-x0)**2/2.0/width_array[cnt]**2)
                    for js in range(ns):
                        for ks in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,ks,cnt] = sum(dist*y[iorb,jorb,js,ks])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1
        return ynew
    
    def BgaussianLatDyn(self,x,y,w1,temperature,cutoff):
        norb = y.shape[0]
        ns = y.shape[2]
        nk = y.shape[3]
        nft = y.shape[4]
        ynew = np.zeros((norb,norb,ns,ns,nk,nft),dtype=complex,order='F')
        for ik in range(nk):
            ynew[...,ik,:] = self.BgaussianLocDyn(x,y[...,ik,:],w1,temperature,cutoff)
        return ynew
    
    def Polarizability(self, Gf : np.ndarray, FT, rkgrid : list = None) -> np.ndarray:

        '''
        input : G(k,tau), tau, freq, rkgrid <- option
        output : P(R,tau)
        it save the Polarizability with k, frequency domain in BHamiltonian
        '''
        Gf = self.fermion.FLatDyn_K2R(self.rkgrid,Gf)
        norbc = Gf.shape[0]
        ns = self.ns
        nr = Gf.shape[3]
        ntau = Gf.shape[4]
        norb = len(self.bind)
        tau = FT.tau
        nu = FT.nu

        if rkgrid == None:
            rkgrid = self.rkgrid
        else:
            rkgrid = rkgrid

        tempmat = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nr,ntau),dtype=complex,order='F')
        Pol = np.zeros((norb,norb,ns,ns,nr,ntau),dtype=complex,order='F')

        Gf_m = self.fermion.FLatTau_m(Gf)

        if self.ns == 2:
            for itau in range(ntau):
                for ir in range(nr):
                    for js in range(ns):
                        for ks in range(ns):
                            for iorbc, jorbc,korbc,lorbc in itertools.product(list(range(norbc)),list(range(norbc)),list(range(norbc)),list(range(norbc))):
                                if js == ks:
                                    tempmat[iorbc,lorbc,jorbc,korbc,js,ks,ir,itau] = Gf_m[jorbc,iorbc,js,ir,itau]*Gf[korbc,lorbc,ks,ir,itau]
        else:
            if self.SOC == True:
                C = 1
                for itau in range(ntau):
                    for ir in range(nr):
                        for iorbc, jorbc,korbc,lorbc in itertools.product(list(range(norbc)),list(range(norbc)),list(range(norbc)),list(range(norbc))):
                            tempmat[iorbc,lorbc,jorbc,korbc,0,0,ir,itau] = Gf_m[jorbc,iorbc,0,ir,itau]*Gf[korbc,lorbc,0,ir,itau]*C
            else:
                C = 2
                for itau in range(ntau):
                    for ir in range(nr):
                        for iorbc, jorbc,korbc,lorbc in itertools.product(list(range(norbc)),list(range(norbc)),list(range(norbc)),list(range(norbc))):
                            tempmat[iorbc,lorbc,jorbc,korbc,0,0,ir,itau] = Gf_m[jorbc,iorbc,0,ir,itau]*Gf[korbc,lorbc,0,ir,itau]*C
        
        Pol = self.Convert_4_2_LatDyn(tempmat,1)
        # Pol_kt = self.BLatStc_R2K(self.rkgrid,Pol)
        # Pol_kf = self.BLatDyn_T2F(tau,Pol_kt,nu)
        self.Pol = Pol # P(k,f)


    def dielectric_function(self,Pol : np.ndarray, V : np.ndarray) -> np.ndarray:

        norb = Pol.shape[0]
        ns = self.ns
        nk = Pol.shape[4]
        nnu = Pol.shape[5]
        
        epsilon = np.zeros((norb,norb,ns,ns,nk,nnu),dtype=complex,order='F')


        epsilon = self.BImMLatDyn(Pol,V)

        return epsilon
    
    def inv_dielectric_function(self,Pol : np.ndarray, V : np.ndarray) -> np.ndarray:

        norb = Pol.shape[0]
        ns = self.ns
        nk = Pol.shape[4]
        nnu = Pol.shape[5]

        epsilon_inv = np.zeros((norb,norb,ns,ns,nk,nnu),dtype=complex,order='F')

        epsilon_inv = self.BIimMLatDyn(Pol,V)

        return epsilon_inv
    
    def screened_coulomb(self, Pol : np.ndarray, V : np.ndarray) -> np.ndarray:

        '''
        input P(k,f), V(k)
        return W(k,f)
        
        it save the screened coulomb interaction with k frequency domain in BHamiltonian
        '''

        norb = Pol.shape[0]
        ns = self.ns
        nk = Pol.shape[4]
        nnu = Pol.shape[5]
        norbc = len(self.fermion.find)

        tempmat = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nnu),dtype=complex,order='F')
        W = np.zeros((norb,norb,ns,ns,nk,nnu),dtype=complex,order='F')
        V_dyn = np.zeros((norb,norb,ns,ns,nk,nnu),dtype=complex,order='F')

        for inu in range(nnu):
            V_dyn[:,:,:,:,:,inu] = V[:,:,:,:,:]

        Pol_comp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nnu),dtype=complex,order='F')
        V_comp = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nnu),dtype=complex,order='F')
        epsilon_inv = np.zeros((norbc*norbc,norbc*norbc,ns,ns,nk,nnu),dtype=complex,order='F')

        Pol_comp = self.full_2_2LatDyn(Pol,0)
        V_comp = self.full_2_2LatDyn(V_dyn,0)
        tempmat = DiagE.dyson.blatdyn(V_comp,Pol_comp) #-> check
        # epsilon_inv = self.inv_dielectric_function(Pol_comp,V_comp)
        
        # tempmat = self.BMulLatDyn(V_comp,epsilon_inv)


        W = self.full_2_2LatDyn(tempmat,1)
        
        self.W = W

    
    def screened_coulomb_C(self,W : np.ndarray = None, V : np.ndarray = None) -> np.ndarray:

        norb = W.shape[0]
        ns = self.ns
        nk = W.shape[4]
        nnu = W.shape[5]

        
        Wc = np.zeros((norb,norb,ns,ns,nk,nnu),dtype=complex,order='F')
        
        for inu in range(nnu):
            Wc[:,:,:,:,:,inu] = W[:,:,:,:,:,inu] - V[:,:,:,:,:]
        
        self.Wc = Wc


class FT_grid(object):

    def __init__(self,beta : float = None,size : int = None):

        self.beta = beta
        self.size = size
        self.omega = np.zeros((size),dtype=float,order='F')
        self.nu = np.zeros((size),dtype=float,order='F')
        self.tau = np.zeros((size),dtype=float,order='F')

    def Omega(self) -> np.ndarray:

        nomega = self.size
        for iomega in range(nomega):
            self.omega[iomega] = np.pi/self.beta*(2*iomega+1)

    def Tau(self) -> np.ndarray:

        ntau = self.size
        for itau in range(ntau):
            itheta = DiagE.common.ttind(itau,ntau)
            self.tau[itau] = self.beta/2.0*(np.cos(np.pi*(itheta+0.5)/ntau)+1.0)

    def Nu(self) -> np.ndarray:

        nnu = self.size
        for inu in range(nnu):
            self.nu[inu] = np.pi/self.beta*(2*inu)


class Impurity(object):

    def __init__(self,fermion : FHamiltonian,boson : BHamiltonian,FT : FT_grid):
        self.fermion = fermion
        self.boson = boson
        self.FT = FT
        self.prob_space = {}
        self.E_loc = None
        self.E_imp = None
        self.hyb_loc = None
        self.hyb_imp = None
        self.G_loc = None
        self.G_imp = None
        self.Sigma_loc = None
        self.Sigma_imp = None
        
        

    def projector(self,ind_dict : dict): # input : {"1":[[0,0],[0,1]]} -> optional : [0,1]
        
        
        nspace = 0
        forbc = 0
        
        prob_space = {}
        for key, val in ind_dict.items():
            # old = nspace
            self.prob_space[key]=[]
            for orb_list in val:
                atom = 0
                for orb in orb_list:
                    if orb == orb_list[0]:
                        atom = orb[0]
                    if atom != orb[0]:
                        print("Different atoms are involved in the same space")
                        sys.exit()
                    
            # for ii in range(old,nspace):

            #     self.prob_space[key].append(ii+1)
            prob_space[key] = [nspace+i for i in range(len(val))]
            nspace += len(val)
        # print(prob_space)
        ns = self.boson.ns
        self.fermion.prob_space = prob_space
        self.boson.prob_space = prob_space
        
        
        
        new_dict = {}
        for key, val in ind_dict.items():
            new_dict[key] = []
            for ind in val:
                temp_list = []
                for orb in ind:
                    find = self.fermion.Findex(orb)
                    temp_list.append(find)
                new_dict[key].append(temp_list)
        for key,val in new_dict.items():
            for orb in val:
                if len(orb)>=forbc:
                    forbc=len(orb)
        fprojector = np.zeros((len(self.fermion.find),forbc,ns,nspace),dtype=complex,order='F')
        self.fermion.imp_dict = new_dict
        new_dict2 = {}
        borbc = 0
        for key in new_dict.keys():
            new_dict2[key] = []
            nn = len(new_dict[key])
            for ii in range(nn):
                orb = new_dict[key][ii]
                temp_list= []
                for iorb in orb:
                    for jorb in orb:
                        a, m1 = self.fermion.Fatomorb(iorb)
                        b, m2 = self.fermion.Fatomorb(jorb)
                        if a==b:
                            bind = self.boson.b2f.index([iorb,jorb])
                            temp_list.append(bind)
                new_dict2[key].append(temp_list)
        for key, val in new_dict2.items():
            for orb in val:
                if len(orb)>=borbc:
                    borbc=len(orb)
        # print(borbc)
        self.boson.imp_dict = new_dict2

        for js in range(ns):
            for key, val in self.fermion.prob_space.items():
                for ii,ispace in enumerate(val):
                    for ind in self.fermion.imp_dict[key][ii]:
                        print(ind,self.fermion.imp_dict[key][ii].index(ind),ii,ispace)
                        fprojector[ind,self.fermion.imp_dict[key][ii].index(ind),js,ispace] = 1

        

        # for key, val in self.fermion.imp_dict.items():
        #     for ispace in range(nspace):    
                    # for ind in self.fermion.imp_dict[str(iprob+1)][0]:
                    #     fprojector[ind,self.fermion.imp_dict[str(iprob+1)][0].index(ind),js,iprob] = 1

        

        bprojector = np.zeros((len(self.boson.bind),borbc,ns,nspace),dtype=float,order='F')
        # print(bprojector.shape)

        for js in range(ns):
            for key, val in self.boson.prob_space.items():
                for ii, ispace in enumerate(val):
                    for ind in self.boson.imp_dict[key][ii]:
                        bprojector[ind,self.boson.imp_dict[key][ii].index(ind),js,ispace] = 1
        # for iprob in range(nprob):
        #     for js in range(ns):
        #         for ind in self.boson.imp_dict[str(iprob+1)][0]:
        #             bprojector[ind,self.boson.imp_dict[str(iprob+1)][0].index(ind),js,iprob] = 1


        self.fprojector = fprojector
        self.bprojector = bprojector

        return None
    
class Method():

    def __init__(self,fermion : FHamiltonian, boson : BHamiltonian ,FT : FT_grid, IP : Impurity = None):

        self.fermion = fermion
        self.boson = boson
        self.ft = FT
        self.ip = IP
        

    def Hartree_Fock(self, iter_max : int, Hmat : np.ndarray, N : float, mix : float, rkgrid : list = None):
        '''
        return H_hf, Sigma_H, Sigma_F, density_matrix, mu
        '''
        norbc = Hmat.shape[0]
        ns = Hmat.shape[2]
        nk = Hmat.shape[3]
        if rkgrid == None:
            rkgrid = self.boson.rkgrid
        Vk = self.boson.V_bare
        Vr = self.boson.BLatStc_K2R(rkgrid,Vk)

        tau = self.ft.tau
        ntau = len(tau)
        N *= ns
        flattau_init = np.zeros((norbc,norbc,ns,nk,ntau),dtype=complex,order='F')
        flattau_new = np.zeros((norbc,norbc,ns,nk,ntau),dtype=complex,order='F')
        flattau_init = DiagE.bare.flattau(Hmat,tau)

        n_init = self.fermion.occupation_matrix(flattau_init)

        for iter in range(1,iter_max+1):
            if iter == 1:
                n_old = n_init
                flattau_old = flattau_init
            # Sigma_H = self.fermion.Hartree(Vk,flattau_old,self.boson.b2f)
            # flattau_r = self.fermion.FLatDyn_K2R(rkgrid,flattau_old)
            # Sigma_F = self.fermion.Exchange(Vr,flattau_r,self.boson.b2f)
            # Sigma_F = self.fermion.FLatStc_R2K(rkgrid,Sigma_F)
            self.fermion.Hartree(flattau_old,self.boson)
            self.fermion.Exchange(flattau_old,self.boson)
            Sigma_H = self.fermion.Sigma_H
            Sigma_F = self.fermion.Sigma_F
            Hmat_hf = np.zeros((norbc,norbc,ns,nk),dtype=complex,order='F')


            for ik in range(nk):
                for js in range(ns):
                    for iorbc in range(norbc):
                        for jorbc in range(norbc):
                            Hmat_hf[iorbc,jorbc,js,ik] = Hmat[iorbc,jorbc,js,ik]+Sigma_H[iorbc,jorbc,js,ik]+Sigma_F[iorbc,jorbc,js,ik]
            
            
            num = 0
            mu = self.fermion.root_find_for_hf(N,Hmat_hf,tau)

            for ik in range(nk):
                for js in range(ns):
                    for iorbc in range(norbc):
                        Hmat_hf[iorbc,iorbc,js,ik] -= mu

            flattau_new = DiagE.bare.flattau(Hmat_hf,tau)

            n = self.fermion.occupation_matrix(flattau_new)

            n_new = self.fermion.FmixLatStc(iter,mix,n,n_old)

            diff = abs(n_new-n_old)
            
            # for ik in range(nk):
            #     for js in range(ns):
            #         for iorbc in range(norbc):
            #             for jorbc in range(norbc):
            #                 num += diff[iorbc,jorbc,js,ik]/(norbc*norbc*ns*nk)
            num = diff.max()
            print(f'iteration : {iter} \n Criteria \n {num} \n chemical potential : {mu}')

            if (num <= 2.0e-4):
                print(f"Self-consistency is achived with iteration : {iter}")
                return Hmat_hf, Sigma_H, Sigma_F, n_new, mu
            elif (iter==iter_max):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                return Hmat_hf, Sigma_H, Sigma_F, n_new, mu
            else:
                flattau_old = flattau_new
                n_old = n_new

    def GW_approximation(self, iter_max : int, Hmat : np.ndarray, N : float, mix : float ,rkgrid : list = None):

        #####Start Initialization#####

        norbc = Hmat.shape[0]
        ns = Hmat.shape[2]
        nk = Hmat.shape[3]
        tau = self.ft.tau
        omega = self.ft.omega
        nu = self.ft.nu

        if rkgrid == None:
            rkgrid = self.boson.rkgrid
        
        ntau = len(tau)
        nomega = len(omega)
        nnu = len(nu)

        Vk = self.boson.V_bare
        Vr = self.boson.BLatStc_K2R(rkgrid,Vk)

        G_kt_init = DiagE.bare.flattau(Hmat,tau)
        

        N *= ns

        #####Finish Initialization#####

        #####SCF Loop begin #####
        for iter in range(1,iter_max+1):
            if iter == 1:
                G_kt = G_kt_init
                G_kf_init = self.fermion.FLatDyn_T2F(tau,G_kt_init,omega)
                G_kf = G_kf_init
                Sigma_old = np.zeros((norbc,norbc,ns,nk,nomega),dtype=complex,order='F')
                n_old = self.fermion.occupation_matrix(G_kt_init)
                mu_old = 0
                
            num = 0
            G_rt = self.fermion.FLatDyn_R2K(rkgrid,G_kt)
            self.fermion.Hartree(G_kt,self.boson)
            self.fermion.Exchange(G_kt,self.boson)
            self.boson.Polarizability(G_kt,self.ft)
            Pol_kt = self.boson.BLatDyn_R2K(rkgrid,self.boson.Pol)
            Pol_kf = self.boson.BLatDyn_T2F(tau,Pol_kt,nu)
            

            self.boson.screened_coulomb(Pol_kf,Vk)
            self.boson.screened_coulomb_C(self.boson.W,Vk)

            Wc_rf = self.boson.BLatDyn_K2R(rkgrid,self.boson.Wc)
            Wc_rt = self.boson.BLatDyn_F2T(nu,Wc_rf,tau,1,1)
            self.boson.Wc = Wc_rt

            self.fermion.Correlated_self_energy(G_kt,self.boson,self.ft)
            
            self.fermion.Combine_self_energy()
            # Sigma_kf = np.zeros((norbc,norbc,ns,nk,nomega),dtype=complex,order='F')

            # for iomega in range(nomega):
            #     Sigma_kf[:,:,:,:,iomega] = Sigma_C_kf[:,:,:,:,iomega] + Sigma_H + Sigma_F

            Sigma_new = self.fermion.FmixLatDyn(iter,mix,self.fermion.Sigma,Sigma_old)
            tempmat = abs(Sigma_new-Sigma_old)
            max_index = np.unravel_index(np.argmax(tempmat),Sigma_new.shape)
            ii,jj,kk,ll,ff = max_index
            print(f"Sigma_new : {Sigma_new[ii,jj,kk,ll,ff]}, Sigma_bare : {self.fermion.Sigma[ii,jj,kk,ll,ff]}, Sigma_old : {Sigma_old[ii,jj,kk,ll,ff]}")
            print(f"{ii,jj,kk,ll,ff}, Hartre : {self.fermion.Sigma_H[ii,jj,kk,ll]}, Fock : {self.fermion.Sigma_F[ii,jj,kk,ll]}, Correlation : {self.fermion.Sigma_C[ii,jj,kk,ll,ff]}")
            
            G_full_kf = DiagE.dyson.flatdyn(G_kf_init,Sigma_new)
            # G_full_kf = self.fermion.int_FLatFreq(G_kf_init,Sigma_C_kf)
            
            mu = self.fermion.root_find_for_GW(N,G_full_kf,omega,tau)

            chem = np.zeros((norbc,norbc,ns,nk,nomega),dtype=complex,order='F')

            for iomega in range(nomega):
                for ik in range(ns):
                    for js in range(ns):
                        for iorbc in range(norbc):
                            chem[iorbc,iorbc,js,ik,iomega] = -mu
            
            tempmat = G_full_kf
            G_full_kf = DiagE.dyson.flatdyn(tempmat,chem)
            G_full_kt = self.fermion.FLatDyn_F2T(omega,G_full_kf,tau,1,1)
            n_new = self.fermion.occupation_matrix(G_full_kt)

            check = self.fermion_scf(n_new,n_old)
            print(f"iteration : {iter} \n check : \n {check} \n chemical potenital : \n {mu}")

            # if (abs(num)<=1.0e-2):
            if (check<=0.05)and(abs(mu-mu_old)<=0.01):
                print(f"Self-consistency is achived with iteration : {iter}")
                return G_full_kf, self.fermion.Sigma_H, self.fermion.Sigma_F, self.fermion.Sigma_C, Sigma_new, Pol_kf, self.boson.Wc, mu
            elif (iter==iter_max):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                if check <=1.0e-3:
                    print("chemical potential is not converged")
                if abs(mu-mu_old) <= 0.01:
                    print("Self-consistency is not achieved")
                if (check>=1.0e-3)and(abs(mu-mu_old)>=0.01):
                    print("chemical potential and self-consitency is not achieved")
                return G_full_kf, self.fermion.Sigma_H, self.fermion.Sigma_F, self.fermion.Sigma_C, Sigma_new, Pol_kf, self.boson.Wc, mu
            # if (check_r <= 0 and check_i <= 0):
            #     print(f"Self-consistency is achived with iteration : {iter}")
            #     return G_full_kf, self.fermion.Sigma_H, self.fermion.Sigma_F, self.fermion.Sigma_C, Sigma_new, Pol_kf, self.boson.Wc, mu
            # elif (iter==iter_max):
            #     print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
            #     return G_full_kf, self.fermion.Sigma_H, self.fermion.Sigma_F, self.fermion.Sigma_C, Sigma_new, Pol_kf, self.boson.Wc, mu
            else:
                G_kt = G_full_kt
                G_kf = G_full_kf
                Sigma_old = Sigma_new
                mu_old = mu
                n_old = n_new

    def DMFT(self,iter_max : int, Hmat : np.ndarray, N : float, mix : float ,rkgrid : list = None,time1 = 0, time2 = 0, equiv : np.ndarray = None):
        
        norb = Hmat.shape[0]
        ns = Hmat.shape[2]
        nk = Hmat.shape[3]
        tau = self.ft.tau
        omega = self.ft.omega
        nu = self.ft.nu
        ntau = self.ft.size
        nfreq = self.ft.size
        fprojector = self.ip.fprojector
        bprojector = self.ip.bprojector
        nprob = len(self.fermion.imp_dict)
        norbc = fprojector.shape[1]
        nspace = 0
        for val in self.fermion.prob_space.values():
            nspace += len(val)


        if rkgrid == None:
            rkgrid = self.boson.rkgrid
        
        G_latfreq_init = DiagE.bare.flatfreq(Hmat,omega)
        G_loc_freq_init = np.zeros((norbc,norbc,ns,nfreq,nspace),dtype=complex,order='F') # nspace 
        Sigma_init = np.zeros((norbc,norbc,ns,nfreq,nspace),dtype=complex,order='F')
        # for iprob in range(nprob):
        #     G_loc_freq_init[:,:,:,:,iprob] = DiagE.projection.flatdyn(G_latfreq_init,fprojector[:,:,:,iprob]) # projector : nprob -> nspace
        for ispace in range(nspace):
            G_loc_freq_init[...,ispace] = DiagE.projection.flatdyn(G_latfreq_init,fprojector[...,ispace])
        # G_loc_freq_init = DiagE.projection.flatdyn(G_latfreq_init,fprojector)
        Sigma_init = self.fermion.DC_self_energy(G_loc_freq_init,self.boson,self.ip) # nspace
        # Sigma_init_p = convert(Sigma_init) 
        Sigma_init_p = self.fermion.FimpconvertLocDyn(Sigma_init,0)
        for iter in range(1,iter_max+1):
            if iter == 1:
                Sigma_imp = Sigma_init
                Sigma_imp_p = Sigma_init_p
                mu = 0
                Sigma_emb = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')
                G_latfreq = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')
                for ispace in range(nspace): # nspace
                    Sigma_emb += DiagE.embedding.flatdyn(nk,Sigma_init[...,ispace],fprojector[...,ispace]) # nspace
                for ifreq in range(nfreq):
                    for ik in range(nk):
                        for js in range(ns):
                            for iorbc in range(norbc):
                                Sigma_emb[iorbc,iorbc,js,ik,ifreq] -= mu
                    G_latfreq = DiagE.dyson.flatdyn(G_latfreq_init,Sigma_emb)
                mu_old = mu
                Sigma_emb_old = Sigma_emb
            G_locfreq = np.zeros((norbc,norbc,ns,nfreq,nspace),dtype=complex,order='F') # nspace
            G_imp_p = np.zeros((norbc,norbc,ns,nfreq,nprob),dtype=complex,order='F')
            Sigma_hf_p = np.zeros((norbc,norbc,ns,nfreq,nprob),dtype=complex,order='F')
            
            #print(self.Sigma_imp_p.shape, type(self.Sigma_imp_p))
            # for iprob in range(nprob):
            #     G_locfreq[...,iprob] = DiagE.projection.flatdyn(G_latfreq[...,iprob],fprojector[...,iprob]) # nspace
            for ispace in range(nspace):
                G_locfreq[...,ispace] = DiagE.projection.flatdyn(G_latfreq,fprojector[...,ispace])
            G_locfreq_p = self.fermion.FimpconvertLocDyn(G_locfreq,0) # nprob
            
            self.fermion.Energy_imp(Hmat,mu,fprojector) #nspace 
            
            E_imp_p = self.fermion.FimpconvertLocStc(self.fermion.E_imp,0) #problem
            
            # print(E_imp_p)
            ### nspace -> nprob ###
            for iprob in range(nprob):
            
            
                self.fermion.hybridisation(omega,E_imp_p[...,iprob],G_locfreq_p[...,iprob],Sigma_imp_p[...,iprob]) #nprob 
                hyb_dict = self.fermion.write_dict_LocDyn(equiv,self.fermion.hyb)
                self.write_hyb_json(hyb_dict)
            
                self.write_ctqmc_params(str(iprob+1),self.fermion.E_imp[...,iprob],equiv) #
                    
                self.run_ctqmc()
                self.measure_ctqmc()
                # self.impurity_postprocessing()
                G_imp_p[...,iprob], Sigma_imp_p[...,iprob], Sigma_hf_p[...,iprob] = self.impurity_postprocessing(iprob,iter,equiv)
            Sigma_imp = self.fermion.FimpconvertLocDyn(Sigma_imp_p,1)
            for ispace in range(nspace):
                Sigma_emb += DiagE.embedding.flatdyn(nk,Sigma_imp[...,ispace],fprojector[...,ispace])
            Sigma_emb = self.fermion.FmixLatDyn(iter,mix,Sigma_emb,Sigma_emb_old)
            G_latfreq_old = G_latfreq
            G_latfreq = DiagE.dyson.flatdyn(G_latfreq_init,Sigma_emb)
            mu = self.fermion.root_find_for_GW(N,G_latfreq,omega,tau)
            print(G_latfreq.shape)
            chem = np.zeros((norb,norb,ns,nk,nfreq),dtype=complex,order='F')

            for ifreq in range(nfreq):
                for ik in range(ns):
                    for js in range(ns):
                        for iorbc in range(norb):
                            chem[iorbc,iorbc,js,ik,ifreq] = -mu

            tempmat = G_latfreq
            G_latfreq = DiagE.dyson.flatdyn(tempmat,chem)

            check = self.fermion_scf(G_latfreq,G_latfreq_old)

            if (check<=1.0e-3)and(abs(mu-mu_old)<=1.0e-3):
                print(f"Self-consistency is achived with iteration : {iter}")
                return G_latfreq,Sigma_imp,self.fermion.E_imp,self.fermion.hyb,mu
            elif (iter == iter_max):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                return G_latfreq,Sigma_imp,self.fermion.E_imp,self.fermion.hyb,mu
            else:
                mu_old = mu
                Sigma_emb_old = Sigma_emb

    def write_ctqmc_params(self,key,E_imp : np.ndarray,equiv : np.ndarray):
        
        if self.fermion.SOC is False:
            if self.fermion.ns ==1:
                params = {}
                params["hloc"] = {}
                mu_ctqmc=-np.real(E_imp[0,0,0])
                # print(mu_ctqmc,type(mu_ctqmc))
                E_imp = E_imp[:,:,0]-mu_ctqmc*np.eye(E_imp.shape[0],E_imp.shape[0])
                E_imp = np.array(np.real(E_imp),dtype=float)
                tempmat = np.kron(E_imp,np.ones((2,2)))
                params["hloc"]['one body'] = tempmat.tolist()
                self.boson.get_Uijkl_comctqmc(key)
                params["hloc"]["two body"]=self.boson.U_ctqmc.tolist()
                # params["hloc"]["two body"] = {}
                # params["hloc"]["two body"]["parametrisataion"] = "slater-condon"
                # params["hloc"]["two body"]["F0"]=5.0
                # params["hloc"]["two body"]["F2"]=0.0
                # params["hloc"]["two body"]["F4"]=0.0
                # params["hloc"]["two body"]["approximation"] = "none"
                
                params["partition"]={}
                
                params["partition"]["green basis"]= "matsubara"
                params["partition"]["green bulla"]= True
                params["partition"]["green matsubara cutoff"] = 50
                params["partition"]["occupation susceptibility bulla"]=True
                params["partition"]["occupation susceptibility direct"]=False
                params["partition"]["quantum number susceptibility"] = True
                params["partition"]["susceptibility cutoff"]=50
                params["partition"]["susceptibility tail"]=200
                params["partition"]["quantum numbers"]={}
                tempmat = np.ones(E_imp.shape[0]*2)
                params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                # [1,1,1,1,1,1,1,1,1,1]
                for ii in range(len(tempmat)):
                    if ii < E_imp.shape[0]:
                        tempmat[ii]*= 0.5
                    elif ii >= E_imp.shape[0]:
                        tempmat[ii]*=-0.5
                params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                # [0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
                # params["partition"]["observables"]={}
                # params["partition"]["observables"]["S2"] = {}
                params["partition"]["probabilities"]={}
                params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                params["partition"]["density matrix precise"] = True
                params["partition"]["print eigenstates"] = True
                params["partition"]["print density matrix"]= True
                
                # params["dyn"]={}
                # params["dyn"]["quantum numbers"] = np.ones(E_imp.shape[0]*2).tolist()
                # # [[1,1,1,1,1,1,1,1,1,1]]
                # params["dyn"]["functions"] = "dyn.json"
                # params["dyn"]["matrix"] = [["F0"]]
                params["beta"]=self.ft.beta
                params["complex"] = False
                params["mu"]=mu_ctqmc
                params["hybridisation"]={}
                # tempmat2 = np.kron(equiv,np.ones((2,2)))
                tempmat2 = np.kron(equiv,np.eye(2,2))
                tempmat2 = tempmat2.tolist()
                for ii in range(len(tempmat2)):
                    for jj in range(len(tempmat2)):
                        if tempmat2[ii][jj]==0.0:
                            tempmat2[ii][jj] = ""
                        else:
                            tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

                params["hybridisation"]["matrix"]=tempmat2
                params["hybridisation"]["functions"]="hyb.json"
                params["thermalisation time"]=3 #imp['thermalization_time']
                params["quantum number susceptibility"]=True
                params["occupation susceptibility bulla"]=True        
                params["green bulla"]=True       
                params["density matrix precise"]=False #True 
                params["measurement time"]=10 #imp['measurement_time']
                
                with open('params.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                # print("params.json written", file=self.m_ini.control['h_log'])
            elif self.fermion.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.fermion.SOC is True:
            print("SOC must be False")
            sys.exit()

        return None
    
    def write_hyb_json(self,hyb : dict):

        if self.fermion.SOC is False:
            if self.fermion.ns == 1:
                json_dict = {}
                for key,val in hyb.items():
                    json_dict[key] = {}
                    json_dict[key]['beta'] = self.ft.beta
                    json_dict[key]['real'] = np.real(val[0]).tolist()
                    json_dict[key]['imag'] = np.imag(val[0]).tolist()

                with open('hyb.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
            elif self.fermion.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.fermion.SOC is True:
            print("SOC must be False")
            sys.exit()
        return None
        
    
    def run_ctqmc(self):
        
        #run_cmd = 'mpirun -np 1 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        print(run_cmd)
        
        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def measure_ctqmc(self):
        
        run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/EVALSIM params'

        print(run_cmd)
        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd,shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        print("measure self-energy done")

        return None
    
    def impurity_postprocessing(self, key,iter,equiv): # key -> problem number
    
        print("*******************")
        print("Impurity Postprocessing Strat")
        print("*******************")
        print(f'key : {key}')
        fileobs='./params.obs.json'
        filemeas='./params.meas.json'
        
        obsjson = json.load(open(fileobs))
        obsjson = obsjson['partition']
#        print('iter_string',iter_string)
#        exit()
        
    
        histo_temp=obsjson["expansion histogram"]
    
        histo=np.zeros((np.shape(histo_temp)[0], 2))
        histo[:,0]=np.arange(np.shape(histo_temp)[0])
        histo[:,1]=histo_temp
#        nn=json.load(open(fileobs))["scalar"]["N"][0]
        nn=obsjson["scalar"]["N"]       
        ctqmc_sign=obsjson["sign"]
    
        # histogram
        firstmoment=sum(histo[:,0]*histo[:,1])/sum(histo[:,1])
        secondmoment=sum((histo[:,0]-firstmoment)**2*histo[:,1])/sum(histo[:,1])
#        thirdmoment=sum((histo[:,0]-firstmoment)**3*histo[:,1])/sum(histo[:,1])/secondmoment**(3.0/2.0)
        # print('histogram information for impurity_'+self.m_ini.imp['name'])
        print('first moment',  firstmoment)
        print('second moment', secondmoment)
#        print('third moment',  thirdmoment,                      file=self.m_ini.control['h_log'])
    
        # previous_iter_string='_'.join(map(str,iter_string.split('_')[:-1]))+'_'+str(int(iter_string.split('_')[-1])-1)
        
#        exit()
        green = {}
        for key,val in obsjson["green"].items():
            templist = []
            for ii in range(len(val['function']['real'])):
                templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
            green[key]=templist
        Green = self.fermion.read_dict_LocDyn(equiv,green)
        sigma_bare = {}
        sigma_hf = {}
        for key, val in obsjson["self-energy"].items():
            sigma_hf[key] = complex(val['moments'][0])
            templist = []
            for ii in range(len(val['function']['real'])):
                templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
            sigma_bare[key] = templist
        Sigma_hf = self.fermion.read_dict_LocStc(equiv,sigma_hf)
        Sigma_bare = self.fermion.read_dict_LocDyn(equiv,sigma_bare)
        # Sigma_mix = self.fermion.FmixLocDyn(iter,0.1,Sigma_bare,self.Sigma_imp_p[...,key])
     
        params = json.load(open('./params.json'))
        cutoff = params["partition"]["green matsubara cutoff"]
        # occ=self.read_occ_from_jsonfile(fileobs,"occupation",key)
        # if self.fermion.SOC is False:
        #     if self.fermion.ns == 1:
        #         xij = np.zeros((Green.shape[0]*2,Green.shape[0]*2,1,Green.shape[3]),dtype=complex)
        #         for ii in range(Green.shape[0]*2):
        #             for jj in range(Green.shape[0]*2):
        #                 xij[ii,jj,:,:] = obsjson["occupation-susceptibility-bulla"][str(ii)+"_"+str(jj)]['function']
        

        # if(gimpsmt[0] ==0) :
        #   print("use green bare")
        # else :
        #   green=np.zeros(np.shape(green_bare), dtype='complex')   
    
    
        # for jj in range(nimp_orb):
        #     sigma[:,jj]=self.gaussian_broadening_linear(self.m_ini.control['omega'], sigma_bare[:,jj], 0.05, self.m_ini.imp['temperature'], self.m_ini.imp[key]['green_cutoff'])
        Sigma_bare = self.fermion.FgaussianLocDyn(self.ft.omega,Sigma_bare,0.05,1/self.ft.beta,cutoff)
    
        # if (not self.m_ini.imp[key]['para']):
        #     for jj in range(nimp_orb, nimp_orb*2):
        #        mkey=str(-int(key))
        #        sigma[:,jj]=self.gaussian_broadening_linear(self.m_ini.control['omega'], sigma_bare[:,jj], 0.05, self.m_ini.imp['temperature'], self.m_ini.imp[key]['green_cutoff'])

        # if(gimpsmt[0] ==1) :
        #   for jj in range(nimp_orb):
        #     green[:,jj]=self.gaussian_broadening_linear(self.m_ini.control['omega'], green_bare[:,jj], 0.05, self.m_ini.imp['temperature'], self.m_ini.imp[key]['green_cutoff'])
    
        #   if (not self.m_ini.imp[key]['para']):
        #     for jj in range(nimp_orb, nimp_orb*2):
        #        mkey=str(-int(key))
        #        green[:,jj]=self.gaussian_broadening_linear(self.m_ini.control['omega'], green_bare[:,jj], 0.05, self.m_ini.imp['temperature'], self.m_ini.imp[key]['green_cutoff'])
        Green = self.fermion.FgaussianLocDyn(self.ft.omega,Green,0.05,1/self.ft.beta,cutoff)

        # if(gimpsmt[0] ==0) :
        #   return green_bare, sigma, xij, sigma_hf, occ
        # else :
        #   return green, sigma, xij, sigma_hf, occ
        print("*******************")
        print("Impurity Postprocessing Finish")
        print("*******************")
        return Green, Sigma_bare, Sigma_hf

    def read_sigma_imp(self):
        
        obs = json.load(open("params.obs.json"))

        sigma_imag = []
        sigma_real = []
        
        sigma_real.append(obs['self-energy']['function']['real'])
        sigma_imag.append(obs['self-energy']['function']['imag'])
        
        return sigma_real, sigma_imag

    def fermion_scf(self,g_new : np.ndarray,g_old : np.ndarray):
        
        check=0
        tempmat = abs(g_new-g_old)
        check = tempmat.max()

        # for ifreq in range(nfreq):
        #     for ik in range(ns):
        #         for js in range(ns):
        #             for iorb in range(norb):
        #                 for jorb in range(norb):
        #                     check += tempmat[iorb,jorb,js,ik,ifreq]/(norb*norb*ns*nk*nfreq)
        return check
    
    # def gaussian_broadening_linear(self,x, y, w1, temperature, cutoff):
    #     # broadening starts at the second matsubara points
    #     print(np.shape(x))
    #     print(np.shape(y))
    #     print(x)
    #     print(y)
    #     w0=(1.0-3.0*w1)*np.pi*temperature*8.6173303*10**-5
    #     width_array=w0+w1*x
    #     cnt=0
    #     ynew=np.zeros(len(y), dtype='complex')
    #     for x0 in x:
    #         if (x0>cutoff+(w0+w1*cutoff)*3.0):
    #             ynew[cnt]=y[cnt]
    #         else:
    #             if ((x0>3*width_array[cnt]) and ((x[-1]-x0)>3*width_array[cnt])):
    #                 dist=1.0/np.sqrt(2*np.pi)/width_array[cnt]*np.exp(-(x-x0)**2/2.0/width_array[cnt]**2)
    #                 ynew[cnt]=sum(dist*y)/sum(dist)
    #             else:
    #                 ynew[cnt]=y[cnt]
    #         cnt=cnt+1
    #     return ynew
    
    

    
