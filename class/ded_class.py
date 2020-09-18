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
import green_sub
from scipy.fftpack import fftn, ifftn

class lattice():
    def __init__(self, avec):
        avec=np.array(avec)
        self.avec = avec
        self.bvec=np.zeros((3,3))
        self.vol=np.dot(np.cross(avec[:,0], avec[:,1]), avec[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(avec[:,1], avec[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(avec[:,2], avec[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(avec[:,0], avec[:,1])/self.vol

class kpoint():
    def __init__(self,fname=None, meshgrid=None, karray=None):
        self.fname=fname
        self.meshgrid=meshgrid
        self.karray=karray
        
        if (self.fname is not None):
            self.point=np.loadtxt(self.fname)            
            self.nk=len(self.kpoint)
        elif (self.meshgrid is not None):
            meshgrid=np.array(meshgrid)
            self.nk=meshgrid[0]*meshgrid[1]*meshgrid[2]
            kpoint_temp=np.array(list(itertools.product(np.linspace(-0.5, 0.5, num=meshgrid[2], endpoint=False),  np.linspace(-0.5, 0.5, num=meshgrid[1], endpoint=False), np.linspace(-0.5, 0.5, num=meshgrid[0], endpoint=False))))
            self.kpoint=np.fliplr(kpoint_temp)
            self.nk=len(self.kpoint)
            print(self.kpoint)
        elif (self.karray is not None):
            self.kpoint=karray
            self.nk=len(self.kpoint)            


class omega():
    def __init__(self,beta,cutoff):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)        
        self.cutoff=cutoff
        self.nomega=int(floor((cutoff/np.pi*beta-1)/2.0))+1
        self.point=np.pi/beta*(1+2*np.arange(self.nomega))*1j

class nu():
    def __init__(self,beta,cutoff):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)
        self.cutoff=cutoff
        self.nnu=int(floor((cutoff/np.pi*beta)/2.0))+1
        self.point=np.pi/beta*2*np.arange(self.nnu)*1j            
        
class tau():
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
        
        


class Greens():
    
    def __init__(self,gf,beta,rkpoint,tfpoint,onf,onk,uniformrk,moment=None):
        self.beta=beta
        self.temp=1.0/(8.617333262145×10**(-5)*beta)
        self.norb=np.shape(gf)[0]
        self.uniformrk=uniformrk
        self.moment=moment
        
        if (onf and onk):
            self.kf=True
            self.fun_kf=gf
            self.kpoint=rkpoint
            self.fpoint=tfpoint
            self.nk=np.len(rkpoint)
            self.nf=np.len(tfpoint)                                    
        
        elif ((not onf) and onk):
            self.kt=True            
            self.fun_kt=gf
            self.kpoint=rkpoint
            self.tpoint=tfpoint
            self.nk=np.len(rkpoint)
            self.nt=np.len(tfpoint)                                    
            
            
        elif (onf and (not onk)):
            self.rf=True                        
            self.fun_rf=gf
            self.rpoint=rkpoint
            self.fpoint=tfpoint
            self.nf=np.len(rkpoint)
            self.nf=np.len(tfpoint)                                    
            
            
        elif ((not onf) and (not onk)):
            self.rt=True                                    
            self.fun_rt=gf
            self.rpoint=rkpoint
            self.tpoint=tfpoint
            self.nr=np.len(rkpoint)
            self.nt=np.len(tfpoint)                                    
            
            


class FermionG(Greens):

    def __init__(self):
        Greens.__init__(self,gf,beta,rkpoint,tfpoint,onf,onk,uniformrk,moment=None)
        self.ns=np.shape(gf)[2]


    def fourier_r2k(inmat):
        # exp(-ikr)            
        ourmat=ifftn(inmat,axis=(2,3,4))
        return outmat
            
    def fourier_k2r(inmat):
        # exp(ikr)/N
        ourmat=ifftn(inmat,axis=(2,3,4))/np.shape(inmat)[2]/np.shape(inmat)[3]/np.shape(inmat)[4]
        return outmat
        
    
class SingleptlG(FermionG):

    def __init__(self):
        FermionG.__init__(self)
        if ((self.moment is None) and self.kf):
            self.moment=green_sub.single_ptl_green_moment(self.fun_kf,self.fpoint)
        

                                               
    

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
