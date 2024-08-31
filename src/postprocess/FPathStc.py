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
import subprocess
import copy
import h5py
from core.Crystal import Crystal
from core.FLatStc import FLatStc

class FPathStc(object):
    
    def __init__(self, crystal : Crystal = None, obj : object = None, hdf5file : str = 'glob.h5'):

        if (crystal is not None) and (obj is not None):
            pass
        else:
            if os.path.exists(hdf5file):
                glob = h5py.File(hdf5file)
                ini = glob['input']
                tempcry = ini['Crystal']
                cry = {}
                for key in tempcry.keys():
                    if (type(tempcry[key][()])==bytes):
                        cry[key] = str(tempcry[key][()],'utf-8')
                    else:
                        cry[key] = tempcry[key][()]
                for key in cry.keys():
                    if key=='Basis':
                        cry[key] = eval(cry[key])
                    elif key=='KGrid':
                        cry[key] = eval(cry[key])
                    elif key=='RVec':
                        cry[key] = eval(cry[key])
                    else:
                        cry[key] = cry[key]
                
                crystal = Crystal(Rvec=cry['RVec'],CorF=cry['CorF'],Basis=cry['Basis'],Nspin=cry['NSpin'],SOC=cry['SOC'],Nelec=cry['NElec'],KGrid=cry['KGrid'])
                glob.close()
            else:
                print(f"Error : Check the {self.__class__.__name__} input again")
                sys.exit()

        
        
        self.crystal = crystal
        self.obj = obj
        self.flatstc = FLatStc(crystal=self.crystal)
        
    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nrk = mat.shape[3]

        matinv = np.zeros((norb,norb,ns,nrk),dtype=np.complex64,order='F')

        for irk in range(nrk):
            for js in range(ns):
                matinv[:,:,js,irk] = np.linalg.inv(mat[:,:,js,irk])
        
        return matinv    
    

    def R2K(self,matr : np.ndarray = None,kpoint : np.ndarray = None): # R2KAny

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
                        # for ir in range(nr):
                        #     [a,m1] = self.crystal.FAtomOrb(iorb)
                        #     [b,m2] = self.crystal.FAtomOrb(jorb)
                        #     delta = self.crystal.basisf[a,:]-self.crystal.basisf[b,:]
                        #     temp += tempmat[iorb,jorb,js,ir]*np.exp(-2.0j*np.pi*(kpoint[ik]@(delta-self.crystal.rvec[ir])))
                        # matk[iorb,jorb,js,ik] = temp
                        
        
        return matk
    
    def Gaussian(self, x, mu, sigma=0.1):

        return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    
    def Dos(self, matr : np.ndarray = None, kgrid = [20,20,20], sigma : float = 0.1, plotoption : bool = False, energyrange : list = None):

        print("***** DOS Calculation Start *****")
        norb = matr.shape[0]
        ns = matr.shape[2]
        if (type(kgrid) is list):
            nk = kgrid[0]*kgrid[1]*kgrid[2]
            kpointtemp = np.array(list(itertools.product(np.linspace(0,1,num=kgrid[2],endpoint=False),np.linspace(0,1,num=kgrid[1],endpoint=False),np.linspace(0,1,num=kgrid[0],endpoint=False))))
            kpoint = np.fliplr(kpointtemp)
        elif (type(kgrid) is np.ndarray):
            nk = len(kgrid)
            kpoint = kgrid
        
        print("***** Fourier transfrom R2K Start *****")
        matk = self.R2K(matr=matr,kpoint=kpoint)
        print("***** Fourier transfrom R2K Finish *****")
        print("***** Hamiltonian Diagonalization Start *****")
        (energy, eigvec) = self.flatstc.Diagonalize(matk=matk,eigvec=True)
        print("***** Hamiltonian Diagonalization Finish *****")

        if energyrange == None:
            emin = -10
            emax = 10
        else:
            emin = energy[0]
            emax = energy[-1]
        E = np.linspace(emin,emax,nk)
        dos = np.zeros((norb,ns,nk),dtype=complex)
        tempmat = np.zeros((norb,ns,nk),dtype=float)

        print("***** Gaussian Approach Start *****")
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    e = energy[iorb,iorb,js,ik]
                    tempmat[iorb,js] += self.Gaussian(E,e,sigma)/nk
        print("***** Gaussian Approach Finish *****")

        eiginv = self.Inverse(eigvec)

        for ik in range(nk):
            for js in range(ns):
                D = np.diag(tempmat[:,js,ik])
                tempmat2 = eigvec[:,:,js,ik]@(D@eiginv[:,:,js,ik])
                for iorb in range(norb):
                    dos[iorb,js,ik] = tempmat2[iorb,iorb]

        print(f"Integration gaussian : {np.trapz(self.Gaussian(E,0),E)}")

        temp = 0
        for js in range(ns):
            for iorb in range(norb):
                temp+= np.trapz(dos[iorb,js],E)

        print(f'Integration dos : {temp}')

        if plotoption:
            fig, ax = plt.subplots()
            ax.set_xlim(E[0],E[-1])
            legend = []
            for js in range(ns):
                for iorb in range(norb):
                    ax.plot(E,dos[iorb,js].real)
                    legend.append(iorb+1)

            ax.legend(legend)
            ax.set_xlabel('E (eV)')
            ax.set_ylabel('DOS')
            plt.show()
        else:
            with open("dos.dat", 'w') as f:
                for ie in range(len(E)):
                    for js in range(ns):
                        linedata = [E[ie]]+dos[:,js,ie].real.tolist()
                        line = ' '.join(map(str, linedata))
                        f.write(line+'\n')
        
        return None
    
    def Band(self, hmat : np.ndarray, fn : str = None, plotoption : bool = False, label : list = None):


        energy = self.flatstc.Diagonalize(hmat)
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
