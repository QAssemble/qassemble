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
from scipy.fftpack import fftn, ifftn

import ../module/DiagE

# I need projector, noninteracting_term

class CProjector():
    def __init__(self,projector):
        self.projector=projector

class CNoninteracting():
    def __init__(self,hmat,rkpoint,k_or_r,uniformrk,beta=None,nomega=None):
    # def __init__(self,hmat,rkpoint,k_or_r,uniformrk):        

        self.norb=hmat[0]
        self.ns=hmat[2]
        # self.beta=beta
        # self.nomega=nomega
        
        if (k_or_r is 'k'):
            self.onk=True
            self.onr=False
            self.hmatk=hmat
            self.kpoint=rkpoint
            self.nk=hmat[3]
        elif (k_or_r is 'r'):
            self.onr=True
            self.onk=False            
            self.hmatr=hmat
            self.rpoint=rkpoint
            self.nr=hmat[3]

    def FourierR2K(self):
        if (onr and uniformrk):
            self.hmat_k=green_sub.fermion_mat_fft(self.rkmesh,self.hmat_r, -1, 1.0) # check_sign
        else:
            print("no Fermion function on R space\n")
        return

    def FourierK2R(self):
        if (onk and uniformrk):
            self.hmat_r=green_sub.fermion_mat_fft(self.rkmesh,self.hmat_k, 1, 1.0/self.nrk) # check_sign
        else:
            print("no Fermion function on R space\n")
        return

    def GKO0(self):
        if ((onk) and (beta is not None) and (nomega is not None)):
            self.opoint=np.pi/beta*(1+2*np.arange(self.nomega))*1j
            self.gko0=green_sub.non_interacting_single_ptl_g(self.lat_k,self.opoint)
        else:
            print("not enough information to define noninteracting green's function\n")
        return
            

class CBareCoulomb():
    def __init__(self,coulomb):
        self.coulomb=coulomb

class CFermionProjector(CProjector):
    def __init__(self):
        pass    

class CLattice():
    def __init__(self, avec):
        avec=np.array(avec)
        self.avec = avec
        self.bvec=np.zeros((3,3))
        self.vol=np.dot(np.cross(avec[:,0], avec[:,1]), avec[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(avec[:,1], avec[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(avec[:,2], avec[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(avec[:,0], avec[:,1])/self.vol

class CKpoint():
    def __init__(self,fname=None, meshgrid=None, karray=None):
        self.fname=fname
        self.meshgrid=meshgrid
        self.karray=karray
        
        if (self.fname is not None):
            self.point=np.loadtxt(self.fname)            
            self.nk=len(self.kpoint)
        elif (self.meshgrid is not None):
            meshgrid=np.array(meshgrid)
            self.rkmesh=meshgrid
            self.nk=meshgrid[0]*meshgrid[1]*meshgrid[2]
            kpoint_temp=np.array(list(itertools.product(np.linspace(-0.5, 0.5, num=meshgrid[2], endpoint=False),  np.linspace(-0.5, 0.5, num=meshgrid[1], endpoint=False), np.linspace(-0.5, 0.5, num=meshgrid[0], endpoint=False))))
            self.kpoint=np.fliplr(kpoint_temp)
            self.nk=len(self.kpoint)
            print(self.kpoint)
        elif (self.karray is not None):
            self.kpoint=karray
            self.nk=len(self.kpoint)            


class COmega():
    def __init__(self,beta,cutoff):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)        
        self.cutoff=cutoff
        self.nomega=int(floor((cutoff/np.pi*beta-1)/2.0))+1
        self.point=np.pi/beta*(1+2*np.arange(self.nomega))*1j

class CNu():
    def __init__(self,beta,cutoff):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)
        self.cutoff=cutoff
        self.nnu=int(floor((cutoff/np.pi*beta)/2.0))+1
        self.point=np.pi/beta*2*np.arange(self.nnu)*1j            
        
class CTau():
    def __init__(self,beta,cutoff=None,ntau=None )
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)
        if (cutoff is not None):
            nomega=int(floor((cutoff/np.pi*beta-1)/2.0))+1            
            ntau=nomega*6
        self.ntau=ntau
            
        tpoint=np.zeros(ntau+1, dtype=float64)
        for itau in range(ntau+1):
            if (itau <=int(ntau/2)):
                tpoint[itau]=(itau/(ntau/2.0))**3/2.0*beta
            else:
                tpoint[itau]=(beta-tau[ntau-itau])        
        self.tpoint=tpoint

        # if (self.fname is not None):
        #     self.kpoint=np.loadtxt(self.fname)            
        #     self.nk=len(self.kpoint)
        # elif (self.meshgrid is not None):
        #     meshgrid=np.array(meshgrid)
        #     self.nk=meshgrid[0]*meshgrid[1]*meshgrid[2]
        #     kpoint_temp=np.array(list(itertools.product(np.linspace(-0.5, 0.5, num=meshgrid[2], endpoint=False),  np.linspace(-0.5, 0.5, num=meshgrid[1], endpoint=False), np.linspace(-0.5, 0.5, num=meshgrid[0], endpoint=False))))
        #     self.kpoint=np.fliplr(kpoint_temp)
        #     self.nk=len(self.kpoint)
        #     print(self.kpoint)
        # elif (self.karray is not None):
        #     self.kpoint=karray
        #     self.nk=len(self.kpoint)            
            
                        

    # nomega=read_nomega(dirname)
    # ntau=nomega*6
    # return ntau
        
        


class CGreens():
    
    def __init__(self,gf,beta,rkpoint,tfpoint,onf,onk,uniformrk,moment=None):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)
        self.norb=np.shape(gf)[0]
        self.uniformrk=uniformrk
        self.moment=moment
        
        if (onf and onk):
            self.onkf=True
            self.lat_kf=gf
            self.kpoint=rkpoint
            self.fpoint=tfpoint
            self.nk=np.len(rkpoint)
            self.nf=np.len(tfpoint)                                    
        
        elif ((not onf) and onk):
            self.onkt=True            
            self.lat_kt=gf
            self.kpoint=rkpoint
            self.tpoint=tfpoint
            self.nk=np.len(rkpoint)
            self.nt=np.len(tfpoint)                                    
            
            
        elif (onf and (not onk)):
            self.onrf=True                        
            self.lat_rf=gf
            self.rpoint=rkpoint
            self.fpoint=tfpoint
            self.nf=np.len(rkpoint)
            self.nf=np.len(tfpoint)                                    
            
            
        elif ((not onf) and (not onk)):
            self.onrt=True                                    
            self.lat_rt=gf
            self.rpoint=rkpoint
            self.tpoint=tfpoint
            self.nr=np.len(rkpoint)
            self.nt=np.len(tfpoint)                                    
            
            
class CNonHamiltonian():


class CFermionG(Greens):

    def __init__(self, projector=None):
        Greens.__init__(self,gf,beta,rkpoint,tfpoint,onf,onk,uniformrk,moment=None)
        self.ns=np.shape(gf)[2]
        self.projector=projector

    def FourierR2K(self):
        if (onr .and. uniformrk):
            if (onf):
                self.lat_kf=green_sub.fermion_mat_fft(self.rkmesh,self.lat_rf, -1, 1.0) # check_sign
            elif (ont):
                self.lat_kt=green_sub.fermion_mat_fft(self.rkmesh,self.lat_rt, -1, 1.0) # check_sign
        else:
            print("no Fermion function on R space\n")
            return

    def FourierK2R(self):
        if (onk .and. uniformrk):
            if (onf):
                self.lat_rf=green_sub.fermion_mat_fft(self.rkmesh,self.lat_kf, 1, 1.0/self.nrk) # check_sign
            elif (ont):
                self.lat_rt=green_sub.fermion_mat_fft(self.rkmesh,self.lat_kt, 1, 1.0/self.nrk) # check_sign
        else:
            print("no Fermion function on R space\n")
            return



    def FourierT2F(self):
        if (ont):
            if (onk):
                self.lat_kf=green_sub.fermion_mat_tau_to_freq(self.tpoint,self.lat_kt,self.freq)
            elif (onr):
                self.lat_rf=green_sub.fermion_mat_tau_to_freq(self.tpoint,self.lat_rt,self.freq)
        else:
            print("no Fermion function on tau-space space\n")
            return
            
        
    def FourierF2T(self):
        if (onf):
            if (moment is not None):
                if (onk):
                    self.lat_kt=green_sub.fermion_mat_omega_to_tau(self.fpoint,self.lat_kf,self.moment,self.tau)
                # elif (onr):
                #     self.lat_rt=green_sub.fermion_mat_omega_to_tau(self.fpoint,self.lat_rf,self.moment,self.tau)
        else:
            print("no Fermion function on tau-space space\n")
            return


    def Projection(self):
        if (self.onkf and uniformrk):
            self.loc_kt=self.green_sub.fermion_projection(self.lat_kf,self.projector)
        else:
            print("no Fermion function on k-omega-space\n")
            return
        
        
    

class CSinglePtlG(FermionG):
    
    def __init__(self):
        FermionG.__init__(self)
        if ((self.moment is None) and self.onkf):
            self.moment=green_sub.single_ptl_green_moment(self.lat_kf,self.fpoint)
    

                                               
class CSelfE(FermionG):    

    def LGFunInv(self,nonhmat,projector,embeded):
        nk=np.shape(nunhmat)[2]
        norb=np.shape(nunhmat)[1]
        LGFunInv=zeros((norb,norb,nk,nomega), dtype=complex)

        for k, freq in itertools.product(kvec, freq):
            LGFunInv=freq-hmat[:,:,k]
            

        return temp    
        

        
        # def fourier_t2f(inmat):

            

            

        # def fourier_f2t(inmat):            
    

# class BosonG(Greens):

#     def fourier_tau_nu():
#         pass

#     def fourier_nu_tau():
#         pass

#     def nu(self,beta):
#         pass

    
    

    
    
# class 
