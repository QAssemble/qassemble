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
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE



class Crystal():
    def __init__(self,latt : list,basis_position : list):
        latt = np.array(latt,dtype=float)
        basis_position = np.array(basis_position,dtype=float)
        self.lat = latt
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
        k_mat = np.linalg.inv(np.dot(self.lat,self.lat.T))
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

    def __init__(self,crystal=None, ns : int = None, SOC : bool = False):

        self.ns = ns
        self.basis_f = crystal.basis_f
        self.basis_c = crystal.basis_c
        self.latt = crystal.lat
        self.bvec = crystal.bvec
        self.kpoint = crystal.kpoint
        self.kpath = crystal.kpath
        self.vol = crystal.vol
        self.grid = crystal.grid
        self.SOC = SOC
        self.idx = {}
        self.orbidx = {}
        self.b2f = []
        self.Hopping = []
        self.rvec = []
        self.On_site = []
        self.Ham_R = None

    def set_basis_index(self,option : list)->dict:
        idx = []
        for m1 in range(option[1]):
            idx.append([option[0],m1])
        norbc = len(self.idx)
        ind = 0
        for iorb in range(norbc,norbc+option[1]):
            self.idx[iorb] = idx[ind]
            ind +=1
        idx1 = []
        orb_ind = []
        orb_ind = list(range(option[1]))
        for m1, m2 in itertools.product(orb_ind,orb_ind):
            idx1.append([option[0],[m1,m2]])


        norb = len(self.orbidx)
        ind = 0
        for iorb in range(norb,norb+option[1]**2):
            self.orbidx[iorb] = idx1[ind]
            ind +=1
            self.boson2fermion(iorb)
        
        norb = len(self.orbidx)
        norbc = len(self.idx)

        self.Ham_R = np.zeros((norb,norb),dtype=complex,order='F')
        
        self.V_loc = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
        
    
    def boson2fermion(self,iorb):
        
        
        [a,[m1,m2]] = self.orbkey2val(iorb)
        iorbc1 = self.val2key([a,m1])
        iorbc2 = self.val2key([a,m2])
        self.b2f.append([iorbc1,iorbc2])

        

    def key2val(self,key : int = None) -> list:
        return self.idx[key]

    def val2key(self,val : list = None) -> int :

        for key, value in self.idx.items():
            if value == val:
                return key

    def orbkey2val(self,key : int = None) -> list:
        return self.orbidx[key]

    def orbval2key(self,val : list = None) -> int :
        for key, value in self.orbidx.items():
            if value==val:
                return key

#    def block_val(self,val : complex):

#        if self.ns == 1:
#            return val
#        elif self.ns == 2:
#            val_temp = np.zeros((self.ns),dtype=complex)
#            val_arr = np.array(val)

#            if val_arr.shape==():
#                val_temp[0] += val_arr
#                val_temp[1] += val_arr

#                return val_temp


    def Hoppinglist(self,hopping : float = 0, ind_i : int = 0, ind_j : int = 0, sx : int = 0, sy : int = 0, sz : int = 0) -> list:

        
        stride = np.array([sx,sy,sz])


        self.Ham_R[ind_i,ind_j] = hopping
        self.Ham_R[ind_j,ind_i] = hopping
        alpha = self.key2val(ind_i)
        beta = self.key2val(ind_j)

        
        rv = self.basis_f[alpha,:]-self.basis_f[beta,:]+stride
        

        self.rvec.append([ind_i,ind_j,rv[0]])

        self.Hopping.append([hopping,ind_i,ind_j,stride])
        
        

    def On_site_list(self,Energy : list) -> list :

        for iorb, e in enumerate(Energy):
            self.Ham_R[iorb,iorb] = e

    def Hamiltonian(self,option : str = 'mesh') -> np.ndarray:

        if option == 'mesh':
            kvec = self.kpoint
        elif option == 'path':
            kvec = self.kpath
        nk = len(kvec)
        norb = len(self.idx)
        ham = np.zeros((norb,norb,self.ns,nk),dtype=complex,order='F')

        # Ham_R = self.Ham_R/len(self.rvec)
        for js in range(self.ns):
            for iorb,jorb,R in self.rvec:
                phase = np.exp(-2.0j*np.pi*np.dot(kvec,R))
                for ik in range(nk):
                    ham[iorb,jorb,js,ik] += self.Ham_R[iorb,jorb]*phase[ik]
                    ham[jorb,iorb,js,ik] += self.Ham_R[iorb,jorb]*np.conjugate(phase[ik])


        # for hopp in self.Hopping:
        #     amp = hopp[0]
        #     iorb = hopp[1]
        #     jorb = hopp[2]
        #     R = hopp[3]
            
        #     [a,m1] = self.key2val(iorb)
        #     [b,m2] = self.key2val(jorb)
            

        #     rvec = self.basis_f[a,:] - self.basis_f[b,:] + R
            

        #     phase = np.exp(-2.0j*np.pi*np.dot(kinput,rvec))
        #     for s1 in range(self.ns):
        #         for ik in range(nk):
        #             ham[iorb,jorb,s1,ik] += amp*phase[ik]
        #             ham[jorb,iorb,s1,ik] += amp*phase[ik].conjugate()

        return ham

    def diagonalization(self,ham : np.ndarray,eigvec : bool = False):

        nk = ham.shape[3]
        norb = ham.shape[0]
        E = np.zeros((nk,self.ns,norb),dtype=float)
        evec = np.zeros((norb,norb,self.ns,nk),dtype=complex)

        if eigvec == False:
            for ik in range(nk):
                for js in range(self.ns):
                    E[ik,js,:] = np.linalg.eigvalsh(ham[:,:,js,ik])
            return E
        else:
            for ik in range(nk):
                for js in range(self.ns):
                    (energy,eig_vec) = np.linalg.eig(ham[:,:,js,ik])
                    E[ik,js,:] = energy
                    evec[:,:,js,ik] = eig_vec
            return E, evec

    def visualization(self,energy : np.ndarray,filename : str = None):
        
        if self.grid[2] ==1 :
            norb = energy.shape[2]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            kx = self.kpoint[:,0].reshape(self.grid[0],self.grid[1],self.grid[2])
            ky = self.kpoint[:,1].reshape(self.grid[0],self.grid[1],self.grid[2])
            
            energy = energy.reshape(self.grid[0],self.grid[1],self.grid[2],self.ns,norb)

            for js in range(self.ns):
                for iorb in range(norb):
                    ax.plot_surface(kx[:,:,0],ky[:,:,0],energy[:,:,0,js,iorb])
            
            ax.view_init(azim=-120,elev=0)
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Energy eV')
            plt.show()
            if filename is not None:
                fig.savefig(filename)
        elif self.grid[2] is not 1:
            print('Error, kz must be 1')
            sys.exit()
        

    def band(self,energy : np.ndarray):

        import matplotlib.pyplot as plt

        if self.ns == 1:
            plot_energy = energy[:,0,:]
            plt.plot(plot_energy)
        else:
            plot_energy_up = energy[:,0,:]
            plot_energy_down = energy[:,1,:]

            plt.plot(plot_energy_up,'k-')
            plt.plot(plot_energy_down,'r-')
            
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
                [a,m1] = self.key2val(iorb)
                [b,m2] = self.key2val(jorb)

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

                [a,m1] = self.key2val(iorb)
                [b,m2] = self.key2val(jorb)

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

        for ik in range(nk):
            ff[:,:,:,ik,:]  = self.FLocDyn_T2F(tau,ftau[:,:,:,ik,:],freq)
        
        return ff
    
    def FLocDyn_F2T(self, omega : np.ndarray = None, ff : np.ndarray = None, tau : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:

        momentum, high = self.FLocDyn_M(omega,ff,isgreen,highzero)

        ftau = np.empty_like(ff,dtype=complex,order='F')
        ftau = DiagE.fourier.flocdyn_f2t(omega,ff,momentum,tau)

        return ftau
    
    def FLatDyn_F2T(self, omega : np.ndarray = None, ff : np.ndarray = None, tau : np.ndarray = None, isgreen : int = None, highzero : int = None) -> np.ndarray:
        
        nk = ff.shape[3]
        ftau = np.empty_like(ff,dtype=complex,order='F')

        for ik in range(nk):
            ftau[:,:,:,ik,:] = self.FLocDyn_F2T(omega,ff[:,:,:,ik,:],tau,isgreen,highzero)

        return ftau
    
    def FProjLocStc(self,ff : np.ndarray, projector : np.ndarray)->np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        norbc = projector.shape[0]

        ffc = np.zeros((norbc,norbc,ns),dtype=complex,order='F')

        ffc = DiagE.projection.flocstc(ff,projector)

        return ffc
    
    def FProjLatStc(self,ff : np.ndarray, projector : np.ndarray)->np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        norbc = projector.shape[0]

        ffc = np.zeros((norbc,norbc,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            ffc[:,:,:,ik] = self.FProjLocStc(ff[:,:,:,ik],projector[:,:,:,ik])
    
        return ffc
    
    def FProjLocDyn(self,ff : np.ndarray, projector : np.ndarray)->np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        norbc = projector.shape[0]
        nt = ff.shape[3]

        ffc = np.zeros((norbc,norbc,ns,nt),dtype=complex,order='F')

        for it in range(nt):
            ffc[:,:,:,it] = self.FProjLocStc(ff[:,:,:,it],projector[:,:,:,it])

        return ffc
    
    def FProjLatDyn(self,ff : np.ndarray, projector : np.ndarray)->np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        nk = ff.shape[3]
        nt = ff.shape[4]
        norbc = projector.shape[0]

        ffc = np.zeros((norbc,norbc,ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            ffc[:,:,:,:,it] = self.FProjLatStc(ff[:,:,:,:,it],projector[:,:,:,:,it])
        
        return ffc

    
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
        nk = G.shape[3]
        nomega = G.shape[4]

        chem = np.empty_like(G,dtype=complex,order='F')
        G_cal = np.empty_like(G,dtype=complex,order='F')
        tempmat = np.empty_like(G,dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    tempmat[:,:,js,ik,iomega] = np.linalg.inv(G[:,:,js,ik,iomega])
                    for iorb in range(norb):
                        chem[iorb,iorb,js,ik,iomega] = mu
        
        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            tempmat[iorb,jorb,js,ik,iomega] = tempmat[iorb,jorb,js,ik,iomega] + chem[iorb,jorb,js,ik,iomega]

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    G_cal[:,:,js,ik,iomega] = np.linalg.inv(tempmat[:,:,js,ik,iomega])
 
        G_cal = self.FLatDyn_F2T(omega,G_cal,tau,1,1)

        ntau = G_cal.shape[4]
        Ne = 0
        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                    Ne += -G_cal[iorb,iorb,js,ik,ntau-1]

        return Nt-Ne/nk
    
    def root_find_for_GW(self,Nt : float, G : np.ndarray, omega : np.ndarray, tau : np.ndarray):
        
        mu_min = -40
        mu_max = 40

        sol = scipy.optimize.bisect(self.num_of_e_freq,mu_min,mu_max,args=(Nt,G,omega,tau))

        return sol
    
    def occupation_matrix(self, flattau : np.ndarray = None):
        '''
        input : G(k,tau)
        output : occupancy matrix
        '''

        
        norb = flattau.shape[0]
        nk = flattau.shape[3]
        ntau = flattau.shape[4]
        nmat = np.zeros((norb,norb,self.ns),dtype=float)

        for ik in range(nk):
            for js in range(self.ns):
                for iorb in range(norb):
                    nmat[iorb,iorb,js] += -flattau[iorb,iorb,js,ik,ntau-1]
        nmat /= nk
        return nmat
    
    def Hartree(self, Vinput : np.ndarray, Gnot : np.ndarray) -> np.ndarray:
        '''
        input : Vinput(k,tau), Gnot(k,tau)
        output : Energy(k,tau)
        '''

        nk = Gnot.shape[3]
        ntau = Gnot.shape[4]
        norb = Vinput.shape[0]

        norbc = len(self.idx)
        # 3**2 + 5**2 -> norb norbc -> 3+5
        tempmat = np.zeros((norb*self.ns,norb*self.ns,nk),dtype=complex,order='F')

        Energy = np.zeros((norbc,norbc,self.ns,nk),dtype=complex)

        if (self.ns is not 1):
            for ik in range(nk):
                for s1 in range(self.ns):
                    for iorb in range(norb):
                        nn1 = [iorb,s1]
                        ind1, nn1 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn1)
                        [a,[m1,m2]] = self.orbkey2val(iorb)
                        iorb1 = self.val2key([a,m1])
                        iorb2 = self.val2key([a,m2])
                        for s2 in range(self.ns):
                            for jorb in range(norb):
                                nn2 = [jorb,s2]
                                ind2, nn2 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn2)
                                [b,[m3,m4]] = self.orbkey2val(jorb)
                                iorb3 = self.val2key([b,m3])
                                iorb4 = self.val2key([b,m4])
                                Gf_temp = np.zeros((norbc,norbc,self.ns),dtype=complex)
                                for jk in range(nk):
                                    Gf_temp[iorb4,iorb3,s2] += Gnot[iorb4,iorb3,s2,jk,-1]
                                tempmat[ind1,ind2,ik] = Vinput[iorb,jorb,s1,s2,ik]
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
                        [a,[m1,m2]] = self.orbkey2val(iorb)
                        iorb1 = self.val2key([a,m1])
                        iorb2 = self.val2key([a,m2])
                        if (iorb1==None)or(iorb2==None):
                            continue
                        for jorb in range(norb):
                            [b,[m3,m4]] = self.orbkey2val(jorb)
                            jorb1 = self.val2key([b,m3])
                            jorb2 = self.val2key([b,m4])
                            if (jorb1==None)or(jorb2==None):
                               continue
                            Gf_temp = np.zeros((norbc,norbc,1))
                            for jk in range(nk):
                                Gf_temp[jorb2,jorb1,0] += Gnot[jorb2,jorb1,0,jk,ntau-1]
                            Energy[iorb1,iorb2,0,ik] += -Vinput[iorb,jorb,0,0,0]*1/nk*Gf_temp[jorb2,jorb1,0]*C
            else:
                C = 2
                for ik in range(nk):
                    for iorb in range(norb):
                        [a,[m1,m2]] = self.orbkey2val(iorb)
                        iorb1 = self.val2key([a,m1])
                        iorb2 = self.val2key([a,m2])
                        if (iorb1==None)or(iorb2==None):
                            continue
                        for jorb in range(norb):
                            [b,[m3,m4]] = self.orbkey2val(jorb)
                            jorb1 = self.val2key([b,m3])
                            jorb2 = self.val2key([b,m4])
                            if (jorb1==None)or(jorb2==None):
                               continue
                            Gf_temp = np.zeros((norbc,norbc,1))
                            for jk in range(nk):
                                Gf_temp[jorb2,jorb1,0] += Gnot[jorb2,jorb1,0,jk,ntau-1]
                            Energy[iorb1,iorb2,0,ik] += -Vinput[iorb,jorb,0,0,0]*1/nk*Gf_temp[jorb2,jorb1,0]*C

        return Energy   
 
    def Exchange(self, Vinput : np.ndarray, Gf : np.ndarray) -> np.ndarray:
        '''
        input : Vinpu(R,tau), Gf(R,tau)
        output : Energy_ex(R,tau)
        '''



        nk = Gf.shape[3]
        ntau = Gf.shape[4]
        norb = Vinput.shape[0]



        
        Gf = Gf[:,:,:,:,ntau-1]
    

        norbc = len(self.idx)
        Energy = np.zeros((norbc,norbc,self.ns,nk))
        
        for ik in range(nk):
            for s1 in range(self.ns):
                for iorb in range(norb):
                    [a,[m1,m4]] = self.orbkey2val(iorb)
                    iorb1 = self.val2key([a,m1])
                    iorb2 = self.val2key([a,m4])
                    if (iorb1==None)or(iorb2==None):
                        continue
                    for jorb in range(norb):
                        [b,[m2,m3]] = self.orbkey2val(jorb)
                        jorb1 = self.val2key([b,m2])
                        jorb2 = self.val2key([b,m3])
                        if (jorb1==None)or(jorb2==None):
                            continue
                        Energy[iorb1,jorb1,s1,ik] = Gf[iorb2,jorb2,s1,ik] * Vinput[iorb,jorb,s1,s1,ik]


        return Energy
    
    def Correlated_self_energy(self, Wc : np.ndarray = None, Gf : np.ndarray = None) -> np.ndarray:
        
        nr = Gf.shape[3]
        ntau = Gf.shape[4]
        norb = Wc.shape[0]
        norbc = Gf.shape[0]

        Energy = np.zeros((norbc,norbc,self.ns,nr,ntau),dtype=complex,order='F')
        
        for itau in range(ntau):
            for ir in range(nr):
                for s1 in range(self.ns):
                    for s2 in range(self.ns):
                        for iorb in range(norb):
                            [a,[m1,m4]] = self.orbkey2val(iorb)
                            iorb1 = self.val2key([a,m1])
                            iorb2 = self.val2key([a,m4])
                            if (iorb1==None)and(iorb2==None):
                                continue
                            for jorb in range(norb):
                                [b,[m2,m3]] = self.orbkey2val(jorb)
                                jorb1 = self.val2key([b,m2])
                                jorb2 = self.val2key([b,m3])
                                if (jorb1==None)and(jorb2==None):
                                    continue
                                if (s1==s2):
                                    Energy[iorb1,jorb1,s1,ir,itau] += Gf[iorb2,jorb2,s1,ir,itau]*Wc[iorb,jorb,s1,s2,ir,itau]
        
        return Energy
    
    def int_FLatFreq(self,G_not : np.ndarray = None, Energy : np.ndarray = None) -> np.ndarray:
        
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
    
    def Stc_Correlated_self_energy(self, Sigma_C : np.ndarray) -> np.ndarray:
        
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

    def z_factor(self, Sigma : np.ndarray, beta : float)-> np.ndarray:

        norb = Sigma.shape[0]
        ns = Sigma.shape[2]
        nk = Sigma.shape[3]
        nomega = Sigma.shape[4]

        Z = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        tempmat = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        I = np.zeros((norb,norb,ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    I[:,:,js,ik,iomega] = np.eye(norb,norb,dtype=complex,order='F')
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
    
    def QP_Hamiltonian(self,H_not : np.ndarray = None, Hartree : np.ndarray = None, Fock : np.ndarray = None, GW_self : np.ndarray = None, mu : float = None, Z : np.ndarray = None) -> np.ndarray:

        norb = H_not.shape[0]
        ns = H_not.shape[2]
        nk = H_not.shape[3]

        H_QP = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')
        # Chem = np.zeros()
        tempmat = np.zeros((norb,norb,ns,nk),dtype=complex,order='F') # sqrt of z factor
        tempmat2 = np.zeros((norb,norb,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            for js in range(ns):
                eig_val, eig_vec = np.linalg.eig(Z[:,:,js,ik])
                for iorb in range(norb):
                    if 0<=(eig_val[iorb])<=1:
                        continue
                    else:
                        print("Error: The z-factor was calculated incorrectly. Please rerun the code.")
                        exit_code = 1
                        sys.exit(exit_code)
                D = np.diag(eig_val)
                tempmat[:,:,js,ik] = np.dot(np.dot(eig_vec,np.sqrt(D)),np.linalg.inv(eig_vec))
        
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        tempmat2[iorb,jorb,js,ik] = H_not[iorb,jorb,js,ik] + Hartree[iorb,jorb,js,ik] + Fock[iorb,jorb,js,ik] + GW_self[iorb,jorb,js,ik] 
                        if iorb == jorb:
                            tempmat2[iorb,jorb,js,ik] = H_not[iorb,jorb,js,ik] + Hartree[iorb,jorb,js,ik] + Fock[iorb,jorb,js,ik] + GW_self[iorb,jorb,js,ik] -mu
        
        for ik in range(nk):
            for js in range(ns):
                H_QP[:,:,js,ik] = np.dot(np.dot(tempmat[:,:,js,ik],tempmat2[:,:,js,ik]),tempmat[:,:,js,ik])

        return H_QP


class BHamiltonian(object):

    def __init__(self,crystal=None,ns : int = None, SOC : bool = False):
        self.ns = ns
        self.basis_f = crystal.basis_f
        self.basis_c = crystal.basis_c
        self.latt = crystal.lat
        self.bvec = crystal.bvec
        self.kpoint = crystal.kpoint
        self.rkgrid = crystal.grid
        self.nk = crystal.nk
        self.vol = crystal.vol
        self.SOC = SOC
        self.idx = {}
        self.orbidx = {}
        self.int_term = [] 
        self.b2f = []
        self.full_sub = []
        self.V_loc = None
        self.V_nloc = None
        self.V_bare = None


    def set_basis_index(self,option : list)->dict:
        idx = []
        for m1 in range(option[1]):
            idx.append([option[0],m1])
        orb = len(self.idx)
        ind = 0
        for iorb in range(orb,orb+option[1]):
            self.idx[iorb] = idx[ind]
            ind +=1
        idx1 = []
        orb_ind = []
        orb_ind = list(range(option[1]))
        for m1, m2 in itertools.product(orb_ind,orb_ind):
            idx1.append([option[0],[m1,m2]])


        norb = len(self.orbidx)
        ind = 0
        for iorb in range(norb,norb+option[1]**2):
            self.orbidx[iorb] = idx1[ind]
            ind +=1
            self.boson2fermion(iorb)
        
        norb = len(self.orbidx)
        
        self.V_loc = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
        
    
    def boson2fermion(self,iorb):
        
        
        [a,[m1,m2]] = self.orbkey2val(iorb)
        iorbc1 = self.val2key([a,m1])
        iorbc2 = self.val2key([a,m2])
        self.b2f.append([iorbc1,iorbc2])

        

    def key2val(self,key : int = None) -> list:
        return self.idx[key]

    def val2key(self,val : list = None) -> int :

        for key, value in self.idx.items():
            if value == val:
                return key

    def orbkey2val(self,key : int = None) -> list:
        return self.orbidx[key]

    def orbval2key(self,val : list = None) -> int :
        for key, value in self.orbidx.items():
            if value==val:
                return key

    def Four2Two(self,mat : np.ndarray = None) -> np.ndarray:

        norbc = mat.shape[0]
        norb = len(self.orbidx)

        mat_ret = np.zeros((norb,norb),dtype=complex)

        for iorbc in range(norbc):
            for jorbc in range(norbc):
                for korbc in range(norbc):
                    for lorbc in range(norbc):
                        [a,m1] = self.key2val(iorbc)
                        [b,m2] = self.key2val(jorbc)
                        [b_prime,m3] = self.key2val(korbc)
                        [a_prime,m4] = self.key2val(lorbc)

                        if (a==a_prime)and(b==b_prime):
                            iorb = self.orbval2key([a,[m1,m4]])
                            jorb = self.orbval2key([b,[m2,m3]])
                            mat_ret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]
                            
        
        return mat_ret
    
    def Two2Four(self,mat : np.ndarray = None) -> np.ndarray:
        
        norb = mat.shape[0]
        norbc = len(self.idx)

        mat_ret = np.zeros((norbc,norbc,norbc,norbc),dtype=complex)

        for iorb in range(norb):
            for jorb in range(norb):
                [a,[m1,m4]] = self.orbkey2val(iorb)
                [b,[m2,m3]] = self.orbkey2val(jorb)

                iorbc = self.val2key([a,m1])
                jorbc = self.val2key([b,m2])
                korbc = self.val2key([b,m3])
                lorbc = self.val2key([a,m4])
                mat_ret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]
        
        return mat_ret

    def local_interacting(self,option : dict = None) -> np.ndarray:
        '''
        the orbital is fermionic orbital so have to translate to bosoninc orbital
        '''
        norbc = len(option["orbital"])
        ns = self.ns
        

        if option["KorS"]=="K":
            tempmat = self.Kanamori(norbc,option["value"])
            print(tempmat.shape)
            for iorbc in option["orbital"]:
                for jorbc in option["orbital"]:
                    for korbc in option["orbital"]:
                        for lorbc in option["orbital"]:
                            [a,m1] = self.key2val(iorbc)
                            [b,m2] = self.key2val(jorbc)
                            [b_prime,m3] = self.key2val(korbc)
                            [a_prime,m4] = self.key2val(lorbc)
                            if (a==a_prime)and(b==b_prime):
                                iorb = self.orbval2key([a,[m1,m4]])
                                jorb = self.orbval2key([b,[m2,m3]])
                                for s1, s2 in itertools.product(list(range(ns)),list(range(ns))):
                                    self.V_loc[iorb,jorb,s1,s2] = tempmat[m1,m2,m3,m4,s1,s2]
        elif option["KorS"]=="S":
            tempmat = self.Slater_parameter(norbc,option["value"])
            for iorbc in option["orbital"]:
                for jorbc in option["orbital"]:
                    for korbc in option["orbital"]:
                        for lorbc in option["orbital"]:
                            [a,m1] = self.key2val(iorbc)
                            [b,m2] = self.key2val(jorbc)
                            [b_prime,m3] = self.key2val(korbc)
                            [a_prime,m4] = self.key2val(lorbc)
                            if (a==a_prime)and(b==b_prime):
                                iorb = self.orbval2key([a,[m1,m4]])
                                jorb = self.orbval2key([b,[m2,m3]])
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

    def gen_nl_int_ham(self,kgrid : list = None) -> np.ndarray:

        kgrid = np.array(kgrid)
        nk = kgrid[0]*kgrid[1]*kgrid[2]

        kvec = np.array(list(itertools.product(np.arange(0,kgrid[2])/kgrid[2],np.arange(0,kgrid[1])/kgrid[1],np.arange(0,kgrid[0])/kgrid[0])))
        kvec = np.fliplr(kvec)

        norb = len(self.orbidx)

        self.V_nloc = np.zeros((norb,norb,self.ns,self.ns,nk),dtype=complex,order='F')

        for int_term in self.int_term:
            amp = int_term[0]

            iorb = int_term[1]
            jorb = int_term[2]

            [alpha,[m1,m4]] = self.orbkey2val(iorb)
            [beta,[m2,m3]] = self.orbkey2val(jorb)

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
            for js in range(self.ns):
                for ks in range(self.ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            self.V_bare[iorb,jorb,js,ks,ik] = self.V_loc[iorb,jorb,js,ks]+self.V_nloc[iorb,jorb,js,ks,ik]
        
        


    def angular_integral(self,l,k,m1,m2,m3,m4):
        ang_int = 0
        pi = np.pi

        for q in range(-k,k+1):
            ang_int += gaunt(l,k,l,-m1,q,m3)*np.conjugate(gaunt(l,k,l,m4,-q,-m2))*((-1.0 if(m1+q+m2)%2 == 1 else 1.0))

        ang_int *= 4*pi/(2*k+1)

        return ang_int

    def rotaion_matrix(l):

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

        R = BHamiltonian.rotaion_matrix(l)

        R_dag = np.conjugate(np.transpose(R))

        tempmat = np.einsum("ab,cd,bdeg,ef,gh",R_dag,R_dag,V,R,R)

        V = np.array(tempmat,dtype=float,order='F')

        return V
    
    def indexing(self, ntot, ndivision, divisionarray, flag, n1, n2):
        tmpsize = 1
        for size in divisionarray:
            tmpsize *= size

        if tmpsize != ntot:
            print('array_division wrong')
            return

        if flag == 1:
            n1 = n2[0]
            for ii in range(1, ndivision):
                tempcnt = 1
                for jj in range(ii):
                    tempcnt *= divisionarray[jj]
                n1 += (n2[ii] ) * tempcnt
        else:
            n2_array = [0] * ndivision
            tempcnt = n1
            for ii in range(ndivision - 1):
                n2_array[ii] = tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
                tempcnt = (tempcnt - n2_array[ii])//divisionarray[ii]
            n2_array[ndivision - 1] = tempcnt

            # Copy the values from the temporary array to the n2 output array
            for i in range(ndivision):
                n2[i] = n2_array[i]

        return n1, n2
    
    def mapping_full_sub(self):
        norbc = len(self.idx)
        ndim = norbc**2
        ndivision = 2
        divisionarray = [norbc,norbc]

        for ind in range(norbc**2):
            nn1 = [0]*2
            new_ind = 0
            c_ind, nn1 = self.indexing(ndim,ndivision,divisionarray,0,ind,nn1)
            # print(norbc,nn1)
            [a,m1] = self.key2val(nn1[0])
            [a_prime,m4] = self.key2val(nn1[1])
            if a==a_prime:
                b_ind = self.orbval2key([a,[m1,m4]])
                if b_ind is not None:
                    self.full_sub.append([b_ind,c_ind])

    def mapping_mR_R(self):
        
        rkvec = self.kpoint

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
    
    def FLatTau_m(self,flattau : np.ndarray) -> np.ndarray:

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
    
    def BMul(self,mat1 : np.ndarray,mat2 : np.ndarray)->np.ndarray:

        norb = mat1.shape[0]

        mat_out = np.zeros((norb,norb),dtype=complex,order='F')

        mat_out = np.dot(mat1,mat2)

        return mat_out
    
    def BMulLocStc(self,mat1 : np.ndarray, mat2 : np.ndarray)->np.ndarray:

        norb = mat1.shape[0]

        mat_out = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')

        for js in range(self.ns):
            for ks in range(self.ns):
                mat_out[:,:,js,ks] = self.BMul(mat1[:,:,js,ks],mat2[:,:,js,ks])

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
    
       
    def BImMLocStc(self,mat1 : np.ndarray, mat2 : np.ndarray) -> np.ndarray:

        norb = mat1.shape[0]

        mat_out = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
        tempmat = np.eye(norb*self.ns,norb*self.ns,dtype=complex)
        I = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')

        for ind1 in range(norb*self.ns):
            nn1 = [0]*2
            ind1,[iorb,js] = self.indexing(norb*self.ns,2,[norb,self.ns],0,ind1,nn1)
            for ind2 in range(norb*self.ns):
                nn2 = [0]*2
                ind2, [jorb,ks] = self.indexing(norb*self.ns,2,[norb,self.ns],0,ind2,nn2)
                I[iorb,jorb,js,ks] = tempmat[ind1,ind2]
        
        tempmat2 = self.BMulLocStc(mat1,mat2)

        for js in range(self.ns):
            for ks in range(self.ns):
                for iorb in range(norb):
                    for jorb in range(norb):
                        mat_out[iorb,jorb,js,ks] = I[iorb,jorb,js,ks]-tempmat2[iorb,jorb,js,ks]
        
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

        mat_out = np.zeros((norb,norb,self.ns,self.ns),dtype=complex,order='F')
        tempmat = np.zeros((norb*self.ns,norb*self.ns),dtype=complex,order='F')
        tempmat2 = np.zeros((norb*self.ns,norb*self.ns),dtype=complex,order='F')
        

        mat_temp = self.BImMLocStc(mat1,mat2)

        for js in range(self.ns):
            for iorb in range(norb):
                nn1 = [iorb,js]
                ind1, nn1 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn1)
                for ks in range(self.ns):
                    for jorb in range(norb):
                        nn2 = [jorb,ks]
                        ind2, nn2 = self.indexing(norb*self.ns,2,[norb,self.ns],1,0,nn2)
                        tempmat[ind1,ind2] = mat_temp[iorb,jorb,js,ks]
        tempmat2 = np.linalg.inv(tempmat)
        
        for ind1 in range(norb*self.ns):
            nn1 = [0]*2
            ind1, [iorb,js] = self.indexing(norb*self.ns,2,[norb,self.ns],0,ind1,nn1)
            for ind2 in range(norb*self.ns):
                nn2 = [0]*2
                ind2, [jorb,ks] = self.indexing(norb*self.ns,2,[norb,self.ns],0,ind2,nn2)
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

        if (norb==len(self.orbidx)):
            for iorb in range(norb):
                for jorb in range(norb):

                    alpha,[m1,m2] = self.orbkey2val(iorb)
                    beta,[m3,m4] = self.orbkey2val(jorb)

                    delta = self.basis_f[alpha,:] - self.basis_f[beta,:]

                    phase = np.exp(2.0j*np.pi*np.dot(rkvec,delta))

                    for ik in range(nrk):
                        hmatk[iorb,jorb,:,:,ik] *= phase[ik]
            hmatr = DiagE.fourier.blatstc_k2r(rkgrid,hmatk)
        else:
            tempmat = hmatk
            hmatk = np.zeros((len(self.orbidx),len(self.orbidx),self.ns,self.ns,nrk),dtype=complex,order='F')
            for js in range(self.ns):
                for ks in range(self.ns):
                    for iorb,ind1 in self.full_sub:
                        for jorb,ind2 in self.full_sub:
                            
                            alpha,[m1,m4] = self.orbkey2val(iorb)
                            beta,[m2,m3] = self.orbkey2val(jorb)

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
                        for iorb, ind1 in self.full_sub:
                            for jorb, ind2 in self.full_sub:
                                hmatr[ind1,ind2,js,ks,ir] = tempmat[iorb,jorb,js,ks,ir]

        

        return hmatr

    def BLatStc_R2K(self,rkgrid : list = None, hmatr : np.ndarray = None) -> np.ndarray:

        rkgrid = np.array(rkgrid,dtype=int,order='F')
        nrk = rkgrid[0]*rkgrid[1]*rkgrid[2]
        norb = hmatr.shape[0]

        hmatk = DiagE.fourier.blatstc_r2k(rkgrid,hmatr)

        rkvec = np.array(list(itertools.product(np.arange(0,rkgrid[2])/rkgrid[2],np.arange(0,rkgrid[1])/rkgrid[1],np.arange(0,rkgrid[0])/rkgrid[0])))
        rkvec = np.fliplr(rkvec)


        if (norb==len(self.orbidx)):
            for iorb in range(norb):
                for jorb in range(norb):

                    alpha,[m1,m2] = self.orbkey2val(iorb)
                    beta,[m3,m4] = self.orbkey2val(jorb)

                    delta = self.basis_f[alpha,:]-self.basis_f[beta,:]

                    phase = np.exp(-2.0j*np.pi*np.dot(rkvec,delta))

                    for ir in range(nrk):
                        hmatk[iorb,jorb,:,:,ir] *= phase[ir]
        else:
            tempmat = hmatk
            hmatk = np.zeros((len(self.orbidx),len(self.orbidx),self.ns,self.ns,nrk),dtype=complex,order='F')

            for js in range(self.ns):
                for ks in range(self.ns):
                    for iorb, ind1 in self.full_sub:
                        for jorb, ind2 in self.full_sub:

                            alpha,[m1,m4] = self.orbkey2val(iorb)
                            beta,[m2,m3] = self.orbkey2val(jorb)

                            delta = self.basis_f[alpha,:] - self.basis_f[beta,:]

                            phase = np.exp(-2.0j*np.pi*np.dot(rkvec,delta))

                            for ik in range(nrk):
                                hmatk[iorb,jorb,js,ks,ik] = tempmat[ind1,ind2,js,ks,ik]*phase[ik]
            
            tempmat = hmatk
            hmatk = np.zeros((norb,norb,self.ns,self.ns,nrk),dtype=complex,order='F')

            for ik in range(nrk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb, ind1 in self.full_sub:
                            for jorb, ind2 in self.full_sub:
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

        for ik in range(nk):
            bf[:,:,:,:,ik,:] = self.BLocDyn_T2F(tau,btau[:,:,:,:,ik,:],freq)

        return bf
    
    def BLocDyn_F2T(self,nu : np.ndarray = None,bnu : np.ndarray = None, tau : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:

        momentum, high = self.BLocDyn_M(nu,bnu,oddzero,highzero)

        btau = np.empty_like(bnu, dtype=complex,order='F')
        btau = DiagE.fourier.blocdyn_f2t(nu,bnu,momentum,tau)

        return btau
    
    def BLatDyn_F2T(self,nu : np.ndarray = None,bnu : np.ndarray = None, tau : np.ndarray = None, oddzero : int = None, highzero : int = None) -> np.ndarray:

        nk = bnu.shape[4]
        btau = np.empty_like(bnu,dtype=complex,order='F')

        for ik in range(nk):
            btau[:,:,:,:,ik,:] = self.BLocDyn_F2T(nu,bnu[:,:,:,:,ik,:],tau,oddzero,highzero)
        
        return btau
    
    def Polarizability(self, Gf : np.ndarray = None) -> np.ndarray:

        norb = Gf.shape[0]
        nr = Gf.shape[3]
        ntau = Gf.shape[4]
        borb = len(self.orbidx)
        
        tempmat = np.zeros((norb,norb,norb,norb,self.ns,self.ns,nr,ntau),dtype=complex,order='F')
        Pol = np.zeros((borb,borb,self.ns,self.ns,nr,ntau),dtype=complex,order='F')

        Gf_m = self.FLatTau_m(Gf)

        
        if self.ns == 2:
            for itau in range(ntau):
                for ir in range(nr):
                    for js in range(self.ns):
                        for ks in range(self.ns):
                            for iorb, jorb, korb, lorb in itertools.product(list(range(norb)),list(range(norb)),list(range(norb)),list(range(norb))):
                                if js == ks:
                                    tempmat[iorb,lorb,jorb,korb,js,ks,ir,itau] = Gf_m[jorb,iorb,js,ir,itau]*Gf[korb,lorb,ks,ir,itau]
        else:
            if self.SOC == True:
                C = 1
                for itau in range(ntau):
                    for ir in range(nr):
                        for js in range(self.ns):
                            for ks in range(self.ns):
                                for iorb, jorb, korb, lorb in itertools.product(list(range(norb)),list(range(norb)),list(range(norb)),list(range(norb))):
                                    tempmat[iorb,lorb,jorb,korb,js,ks,ir,itau] = Gf_m[jorb,iorb,js,ir,itau]*Gf[korb,lorb,ks,ir,itau]*C
            else:
                C = 2
                for itau in range(ntau):
                    for ir in range(nr):
                        for js in range(self.ns):
                            for ks in range(self.ns):
                                for iorb, jorb, korb, lorb in itertools.product(list(range(norb)),list(range(norb)),list(range(norb)),list(range(norb))):
                                    tempmat[iorb,lorb,jorb,korb,js,ks,ir,itau] = Gf_m[jorb,iorb,js,ir,itau]*Gf[korb,lorb,ks,ir,itau]*C
        
        
        for itau in range(ntau):
            for ir in range(nr):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        Pol[:,:,js,ks,ir,itau] = self.Four2Two(tempmat[:,:,:,:,js,ks,ir,itau])
            
        
        return Pol
    
    def dielectric_function(self,Pol : np.ndarray = None, V : np.ndarray = None) -> np.ndarray:

        norb = Pol.shape[0]
        nk = Pol.shape[4]
        nomega = Pol.shape[5]
        norbc = len(self.idx)
        tempmat = np.zeros((norbc,norbc,norbc,norbc,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        V_dyn = np.zeros((norb,norb,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        epsilon = np.zeros((norb,norb,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            V_dyn[:,:,:,:,:,iomega] = V[:,:,:,:,:]

        

        epsilon = self.BImMLatDyn(Pol,V_dyn)


        return epsilon
    
    def inv_dielectric_function(self,Pol : np.ndarray = None, V : np.ndarray = None) -> np.ndarray:

        norb = Pol.shape[0]
        ns = Pol.shape[2]
        nk = Pol.shape[4]
        nomega = Pol.shape[5]

        epsilon_inv = np.zeros((norb,norb,ns,ns,nk,nomega),dtype=complex,order='F')
        V_dyn = np.zeros((norb,norb,ns,ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(ns):
                    for ks in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                V_dyn[iorb,jorb,js,ks,ik,iomega] = V[iorb,jorb,js,ks,ik]
        
        epsilon_inv = self.BIimMLatDyn(Pol,V_dyn)

        return epsilon_inv 

    def screened_coulomb(self, Pol : np.ndarray = None, V : np.ndarray = None) -> np.ndarray:

        norb = Pol.shape[0]
        nk = Pol.shape[4]
        nomega = Pol.shape[5]
        norbc = len(self.idx)
        borb = len(self.orbidx) 
        
        tempmat = np.zeros((norbc**2,norbc**2,self.ns,self.ns,nk,nomega),dtype=complex,order='F')
        W = np.zeros((borb,borb,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        Pol_comp = np.zeros((norbc**2,norbc**2,self.ns,self.ns,nk,nomega),dtype=complex,order='F')
        V_comp = np.zeros((norbc**2,norbc**2,self.ns,self.ns,nk),dtype=complex,order='F')
        V_comp2 = np.zeros((norbc**2,norbc**2,self.ns,self.ns,nk,nomega),dtype=complex,order='F')
        epsilon_inv = np.zeros((norbc**2,norb**2,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb1, iorb2 in self.full_sub:
                            for jorb1, jorb2 in self.full_sub:
                                Pol_comp[iorb2,jorb2,js,ks,ik,iomega] = Pol[iorb1,jorb1,js,ks,ik,iomega]
                                V_comp[iorb2,jorb2,js,ks,ik] = V[iorb1,jorb1,js,ks,ik]
                                V_comp2[iorb2,jorb2,js,ks,ik,iomega] = V[iorb1,jorb1,js,ks,ik]

        

        
        epsilon_inv = self.inv_dielectric_function(Pol_comp,V_comp)

        tempmat = self.BMulLatDyn(V_comp2,epsilon_inv)
        
        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb1, iorb2 in self.full_sub:
                            for jorb1, jorb2 in self.full_sub:
                                W[iorb1,jorb1,js,ks,ik,iomega] = tempmat[iorb2,jorb2,js,ks,ik,iomega]


        return W
    
    def screened_coulomb_C(self,W : np.ndarray = None, V : np.ndarray = None) -> np.ndarray:

        norb = W.shape[0]
        nk = W.shape[4]
        nomega = W.shape[5]

        V_dyn = np.zeros((norb,norb,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                V_dyn[iorb,jorb,js,ks,ik,iomega] = V[iorb,jorb,js,ks,ik]
        Wc = np.zeros((norb,norb,self.ns,self.ns,nk,nomega),dtype=complex,order='F')

        for iomega in range(nomega):
            for ik in range(nk):
                for js in range(self.ns):
                    for ks in range(self.ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                Wc[iorb,jorb,js,ks,ik,iomega] = W[iorb,jorb,js,ks,ik,iomega]-V_dyn[iorb,jorb,js,ks,ik,iomega]

        return Wc
    
    
class FT_grid():

    def __init__(self,beta : float = None,size : int = None):

        self.beta = beta
        self.size = size
        self.omega = np.zeros((size),dtype=complex,order='F')
        self.nu = np.zeros((size),dtype=complex,order='F')
        self.tau = np.zeros((size),dtype=complex,order='F')

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

class Method():

    def __init__(self,fermion, boson,ft):

        self.fermion = fermion
        self.boson = boson
        self.ft = ft
    
    def Self_consistence_Hartree_Fock(self,iter_max : int, Hmat : np.ndarray, Nt : float, rkgrid : list = None):
        
        norbc = Hmat.shape[0]
        ns = self.boson.ns
        nk = Hmat.shape[3]
        if rkgrid == None:
            rkgrid = self.boson.rkgrid
        Vk = self.boson.V_bare
        Vr = self.boson.BLatStc_K2R(rkgrid,Vk)
        tau = self.ft.tau

        Nt *= ns

        flattau = DiagE.bare.flattau(Hmat,tau)

        n_init = self.fermion.occupation_matrix(flattau)

        for iter in range(1,iter_max):
            if iter == 1:
                n_old = n_init
                flattau_old = flattau
            Sigma_H = self.fermion.Hartree(Vk,flattau_old)
            flattau_r = self.fermion.FLatDyn_K2R(rkgrid,flattau_old)
            tempmat = self.fermion.Exchange(Vr,flattau_r)
            Sigma_F = self.fermion.FLatStc_R2K(rkgrid,tempmat)
            Hmat_hf = np.zeros((norbc,norbc,ns,nk),dtype=complex,order='F')
            
            for ik in range(nk):
                for js in range(ns):
                    for iorbc in range(norbc):
                        for jorbc in range(norbc):
                            Hmat_hf[iorbc,jorbc,js,ik] = Hmat[iorbc,jorbc,js,ik] + Sigma_H[iorbc,jorbc,js,ik] + Sigma_F[iorbc,jorbc,js,ik]
            
            num = 0
            mu = self.fermion.root_find_for_hf(Nt,Hmat_hf,tau)
            
            for ik in range(nk):
                for js in range(ns):
                    for iorbc in range(norbc):
                        Hmat_hf[iorbc,iorbc,js,ik] -= mu
            
            flattau_hf = DiagE.bare.flattau(Hmat_hf,tau)
            
            n_new = self.fermion.occupation_matrix(flattau_hf)

            diff = abs(n_new-n_old)

            for js in range(ns):
                for iorbc in range(norbc):
                    num += diff[iorbc,iorbc,js]/(norbc*ns)
            
            print(f'iteration : {iter} \n n_new : \n {n_new} \n n_old : \n {n_old} \n chemical potential : {mu}')

            if (num <= 1.0e-3):
                print(f"Self-consistency is achived with iteration : {iter}")
                return Hmat_hf, Sigma_H, Sigma_F, n_new
            elif (iter==iter_max):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                return Hmat_hf, Sigma_H, Sigma_F, n_new
            else:
                flattau = flattau_hf
                n_old = n_new
        
    def Self_consistence_GW(self,iter_max : int, Hmat : np.ndarray, Nt : float,rkgrid : list = None):

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

        G_not_kt_init = DiagE.bare.flattau(Hmat,tau)
        n_init = self.fermion.occupation_matrix(G_not_kt_init)

        for iter in range(1,iter_max):
            if iter == 1:
                n_old = n_init
                G_kt = G_not_kt_init
            num = 0

            G_rt = self.fermion.FLatDyn_K2R(rkgrid,G_kt)
            Sigma_H = self.fermion.Hartree(Vk,G_kt)
            tempmat = self.fermion.Exchange(Vr,G_rt)
            Sigma_F = self.fermion.FLatStc_R2K(rkgrid,tempmat)
            Pol_rt = self.boson.Polarizability(G_rt)
            Pol_kt = self.boson.BLatDyn_R2K(rkgrid,Pol_rt)
            Pol_kf = self.boson.BLatDyn_T2F(tau,Pol_kt,nu)

            W_kf = self.boson.screened_coulomb(Pol_kf,Vk)
            Wc_kf = self.boson.screened_coulomb_C(W_kf,Vk)

            Wc_rf = self.boson.BLatDyn_K2R(rkgrid,Wc_kf)
            Wc_rt = self.boson.BLatDyn_F2T(nu,Wc_rf,tau,1,1)

            Sigma_C_rt = self.fermion.Correlated_self_energy(Wc_rt,G_rt)
            Sigma_C_kt = self.fermion.FLatDyn_R2K(rkgrid,Sigma_C_rt)
            Sigma_C_kf = self.fermion.FLatDyn_T2F(tau,Sigma_C_kt,omega)

            Sigma_kf = np.zeros((norbc,norbc,ns,nk,nomega),dtype=complex,order='F')

            for iomega in range(nomega):
                for ik in range(nk):
                    for js in range(ns):
                        for iorbc in range(norbc):
                            for jorbc in range(norbc):
                                Sigma_kf[iorbc,jorbc,js,ik,iomega] = Sigma_C_kf[iorbc,jorbc,js,ik,iomega] + Sigma_H[iorbc,jorbc,js,ik] + Sigma_F[iorbc,jorbc,js,ik]
            
            G_kf = self.fermion.FLatDyn_T2F(tau,G_kt,omega)

            G_full_kf = self.fermion.int_FLatFreq(G_kf,Sigma_kf)

            mu = self.fermion.root_find_for_GW(Nt,G_full_kf,omega,tau)

            G_inv_kf = self.fermion.FInvLatDyn(G_full_kf)

            for iomega in range(nomega):
                for ik in range(nk):
                    for js in range(ns):
                        for iorbc in range(norbc):
                            G_inv_kf[iorbc,iorbc,js,ik,iomega] += mu
            
            G_full_kf = self.fermion.FInvLatDyn(G_inv_kf)

            G_full_kt = self.fermion.FLatDyn_F2T(omega,G_full_kf,tau,1,1)

            n_new = self.fermion.occupation_matrix(G_full_kt)

            diff = n_new - n_old

            print(f"iteration : {iter} \n n_new : \n {n_new} \n n_old : \n {n_old} \n chemical potenital : {mu}")

            for js in range(ns):
                for iorbc in range(norbc):
                    num += diff[iorbc,iorbc,js]/(norbc*ns)
            
            if (abs(num)<=1.0e-2):
                print(f"Self-consistency is achived with iteration : {iter}")
                return G_full_kf, Sigma_H, Sigma_F, Sigma_C_kf, Pol_kf, Wc_kf, mu
            elif (iter==iter_max):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                return G_full_kf, Sigma_H, Sigma_F, Sigma_C_kf, Pol_kf, Wc_kf, mu
            else:
                G_kt = G_full_kt
                n_old = n_new

class Impurity(object):

    def __init__(self,boson,FT):
        self.boson = boson
        self.FT = FT
    
    def projectorLocStc(self,ind_list : list):
    
        ns = self.boson.ns
        fprojector = np.zeros((len(self.boson.idx),len(ind_list),ns),dtype=float,order='F')

        for js in range(ns):
            for ind in ind_list:
                fprojector[ind,ind_list.index(ind),js] = 1

        borb_list = []
        for ind1 in ind_list:
            for ind2 in ind_list:
                [a,m1] = self.boson.key2val(ind1)
                [b,m2] = self.boson.key2val(ind2)
                if a == b:
                    ind = self.boson.b2f.index([ind1,ind2])
                    borb_list.append([ind])

        bprojector = np.zeros((len(self.boson.orbidx),len(borb_list),ns,ns),dtype=float,order='F')

        for js in range(ns):
            for ks in range(ns):
                for borb in borb_list:
                    bprojector[borb,borb_list.index(borb),js,ks] = 1


        return fprojector, bprojector
    
    def projectorLatStc(self,ind_list : list):
        
        nk = len(self.boson.kpoint)

        fprojlocstc, bprojlocstc = self.projectorLocStc(ind_list)
        norb_f = fprojlocstc.shape[0]
        norbc_f = fprojlocstc.shape[1]
        norb_b = bprojlocstc.shape[0]
        norbc_b = bprojlocstc.shape[1]
        ns = fprojlocstc.shape[2]

        fprojlatstc = np.zeros((norb_f,norbc_f,ns,nk),dtype=complex,order='F')
        bprojlatstc = np.zeros((norb_b,norbc_b,ns,ns,nk),dtype=complex,order='F')

        for ik in range(nk):
            fprojlatstc[:,:,:,ik] = fprojlocstc
            bprojlatstc[:,:,:,:,ik] = bprojlocstc
        
        return fprojlatstc, bprojlatstc
    
    def projectorLocDyn(self,ind_list : list):

        nt = self.FT.size
        
        fprojlocstc, bprojlocstc = self.projectorLocStc(ind_list)
        norb_f = fprojlocstc.shape[0]
        norbc_f = fprojlocstc.shape[1]
        norb_b = bprojlocstc.shape[0]
        norbc_b = bprojlocstc.shape[1]
        ns = fprojlocstc.shape[2]

        fprojlocdyn = np.zeros((norb_f,norbc_f,ns,nt),dtype=complex,order='F')
        bprojlocdyn = np.zeros((norb_b,norbc_b,ns,ns,nt),dtype=complex,order='F')

        for it in range(nt):
            fprojlocdyn[:,:,:,it] = fprojlocstc
            bprojlocdyn[:,:,:,:,it] = bprojlocstc
        
        return fprojlocdyn, bprojlocdyn
    
    def projectorLatDyn(self,ind_list : list):

        nt = self.FT.size

        fprojlatstc, bprojlatstc = self.projectorLatStc(ind_list)

        norb_f = fprojlatstc.shape[0]
        norbc_f = fprojlatstc.shape[1]
        norb_b = bprojlatstc.shape[0]
        norbc_b = bprojlatstc.shape[1]
        ns = fprojlatstc.shape[2]
        nk = fprojlatstc.shape[3]

        fprojlatdyn = np.zeros((norb_f,norbc_f,ns,nk,nt),dtype=complex,order='F')
        bprojlatdyn = np.zeros((norb_b,norbc_b,ns,ns,nk,nt),dtype=complex,order='F')

        for it in range(nt):
            fprojlatdyn[:,:,:,:,it] = fprojlatstc
            bprojlatdyn[:,:,:,:,:,it] = bprojlatstc

        return fprojlatdyn, bprojlatdyn

    

