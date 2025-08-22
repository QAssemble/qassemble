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
from .FLatDyn import GreenInt
from .FLocStc import FLocStc,EImp,SigmaHLoc,SigmaFLoc
from .FLatStc import NIHamiltonian,SigmaHartree,SigmaFock
qapath = os.environ.get('QAssemble','')
sys.path.append(qapath+'/src/qacore/modules')
import QAFort

class FLocDyn(object):

    def __init__(self,crystal : Crystal, ft : FTGrid):

        self.crystal = crystal
        self.ft = ft

    def Inverse(self, mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]
        nft = mat.shape[3]
        nprob = mat.shape[4]

        matinv = np.zeros((norb,norb,ns,nft,nprob),dtype=np.complex128,order='F')

        for iprob in range(nprob):
            for ift in range(nft):
                for js in range(ns):
                    matinv[:,:,js,ift,iprob] = np.linalg.inv(mat[:,:,js,ift,iprob])

        return matinv

    def Moment(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]

        moment = np.zeros((norb,norb,ns,3),dtype=np.complex128,order='F')
        high = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        moment, high = QAFort.fourier.flocdyn_m(self.ft.omega,ff,isgreen,highzero)

        return moment, high

    def F2T(self, ff : np.ndarray, isgreen : int, highzero : int) -> np.ndarray:

        norb = ff.shape[0]
        ns = ff.shape[2]
        ntau = len(self.ft.tau)

        ftau = np.zeros((norb,norb,ns,ntau),dtype=np.complex128,order='F')

        moment, high = self.Moment(ff,isgreen,highzero)

        ftau = QAFort.fourier.flocdyn_f2t(self.ft.omega,ff,moment,self.ft.tau)

        return ftau

    def T2F(self,ftau : np.ndarray) -> np.ndarray:

        norb = ftau.shape[0]
        ns = ftau.shape[2]
        nfreq = len(self.ft.omega)

        ff = np.zeros((norb,norb,ns,nfreq),dtype=np.complex128,order='F')

        ff = QAFort.fourier.flocdyn_t2f(self.ft.tau,ftau,self.ft.omega)

        return ff

    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')
        w0 = (1.0 - 3.0*w1)*np.pi*temperature
        widtharray = w0+w1*x
        cnt = 0

        for x0 in x:
            if (x0>cutoff+(w0+w1*cutoff)*3.0):
                ynew[...,cnt]=y[...,cnt]
            else:
                if ((x0>3*widtharray[cnt])and((x[-1]-x0)>3*widtharray[cnt])):
                    dist = 1.0/np.sqrt(2*np.pi)/widtharray[cnt]*np.exp(-(x-x0)**2/2.0/widtharray[cnt]**2)
                    for js in range(ns):
                        for iorb in range(norb):
                            for jorb in range(norb):
                                ynew[iorb,jorb,js,cnt] = sum(dist*y[iorb,jorb,js])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew

    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray):

        norb = Fb.shape[0]
        ns = Fb.shape[2]
        nft = Fb.shape[3]

        Fnew = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        Fnew = mix*Fb+(1.0-mix)*Fold

        return Fnew

    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,nft,nspace),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc

    def Loc2Imp(self,matloc : np.ndarray)->np.ndarray:

        nprob = len(self.crystal.probspace)
        norb = matloc.shape[0]
        ns = matloc.shape[2]
        nft = matloc.shape[3]

        matimp = np.zeros((norb,norb,ns,nft,nprob),dtype=np.complex128,order='F')

        for key, val in self.crystal.probspace.items():
            iprob = int(key)-1
            tempmat = np.zeros((norb,norb,ns,nft),dtype=np.complex128)
            for ispace in val:
                tempmat += matloc[...,ispace]
            tempmat /=len(val)
            matimp[...,iprob] = tempmat

        return matimp
    

    def Loc2Imp_Stc(self,matloc : np.ndarray)-> np.ndarray:

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
                e /=len(pos)
                matdict[ind+1].append(e.tolist())

        return matdict

    def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])

        matout = np.zeros((norb,norb,ns,nfreq),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ind in range(nind):
                pos = self.crystal.FindPositions(equiv,ind+1)
                for ii, jj in pos:
                    matout[ii,jj,js] = matdict[str(ind+1)]

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,nft),dtype=np.complex128,order='F')

        matout = QAFort.dyson.flocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.find)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        matout = np.zeros((norb,norb,ns,nrk,nft),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            matout += QAFort.embedding.flocdyn(nrk,matin[...,ispace],self.crystal.fprojector[...,ispace])

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




    def imp_B2F_freq(self,imp,B):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index

        nprob = len(self.crystal.probspace)
        nft = len(self.ft.omega)

        F = {}
        
        for ii in range(nprob):

            iimp = str(ii+1)

            F[iimp] = {}
            for i in range(len(self.crystal.imp_index[ii])):
                index_of_equivalance = str(i+1)
                
                for k in range(nft):
                    try:
                        F[iimp][index_of_equivalance].append(0)
                    except KeyError:
                        F[iimp][index_of_equivalance] = [0]

                for k in range(nft):
                    for j in range(len(self.crystal.imp_index[ii][i])):
                        F[iimp][index_of_equivalance][k] = F[iimp][index_of_equivalance][k] + B[self.crystal.imp_index[ii][i][j][0],self.crystal.imp_index[ii][i][j][1],k]

                    F[iimp][index_of_equivalance][k] = F[iimp][index_of_equivalance][k]/len(self.crystal.imp_index[ii][i]) ### take the average
            
        return F


    def imp_F2B_freq(self,imp,F):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index
        
        nprob = len(self.crystal.probspace)
        nft = len(self.ft.omega)
        norbc = self.crystal.fprojector.shape[1]

        B = np.zeros((norbc,norbc,nft,nprob),dtype=np.complex128,order='F')

        ii = 0 # index of impurity problems -- nprob
        for key,val in F.items():
            i=0 # index of equivalances
            for valkey,valval in val.items():
                for j in range(len(self.crystal.imp_index[ii][i])):
                    for k in range(nft): # number of omega -- nft
                        B[self.crystal.imp_index[ii][i][j][0],self.crystal.imp_index[ii][i][j][1],k,ii] = valval[k]
                i += 1
            ii += 1

        return B
    

    # def imp_final_input(self,B):

    #     norbc = self.crystal.fprojector.shape[1]
    #     nft = len(self.ft.omega)

    #     I = np.identity(norbc)
    #     mu = np.zeros(nft)
    #     A = np.zeros((norbc,norbc))
    #     A_final = np.zeros((norbc*2,norbc*2,nft))

    #     for ift in range(nft):
    #         mu[ift] = -B[0,0,ift]  ### is the mu the same along the omega space?
    #         A[...] = B[...,ift] + mu[ift]*I
    #         A_final[...,ift] = np.kron(A[...],np.eye(2,2))
        
    #     return A_final,mu


    def write_dict_LocDyn(self,equiv : np.ndarray, mat_in : np.ndarray)->dict:
        
        ns = mat_in.shape[2]
        Nind = int(np.amax(equiv))
        # print(Nind)
        # exit()
        mat_dict = {}    

        for ind in range(Nind):
            mat_dict[ind+1]=[]
            # pos = find_positions(equiv,ind+1)
            pos_row, pos_col = np.where(equiv==ind+1)
            for js in range(ns):
                e = 0
                # for ii, jj in pos:
                for i in range(len(pos_row)):
                    
                    e+=mat_in[pos_row[i],pos_col[i],js]
                e/=len(pos_row)
                mat_dict[ind+1].append(e.tolist())

        return mat_dict
    

    def read_dict_LocDyn(self,equiv : np.ndarray, mat_dict : dict)->np.ndarray:
        
        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(mat_dict["1"])

        mat_out = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')

        Nind = np.amax(equiv)
        
        for js in range(ns):
            for ind in range(Nind):
                # pos = find_positions(equiv,ind+1) 
                pos_row, pos_col = np.where(equiv==ind+1)
                # for ii, jj in pos:
                for i in range(len(pos_row)):
                    mat_out[pos_row[i],pos_col[i],js] = mat_dict[str(ind+1)]
        
        return mat_out
    

    def read_dict_LocStc(self,equiv : np.ndarray, mat_dict : dict)->np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        mat_out = np.zeros((norb,norb,ns),dtype=complex,order='F')

        Nind = np.amax(equiv)
        
        for js in range(ns):
            for ind in range(Nind):
                # pos = find_positions(equiv,ind+1)
                pos_row, pos_col = np.where(equiv==ind+1)
                # for ii,jj in pos:
                for i in range(len(pos_row)):
                    mat_out[pos_row[i],pos_col[i],js] = mat_dict[str(ind+1)]

        return mat_out
    
    def read_susceptibility_LocDyn(self,equiv : np.ndarray, mat_dict : dict)->np.ndarray:
        
        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(mat_dict["1"])

        mat_out = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')

        Nind = np.amax(equiv)
        
        for js in range(ns):
            for ind in range(Nind):
                # pos = find_positions(equiv,ind+1) 
                pos_row, pos_col = np.where(equiv==ind+1)
                # for ii, jj in pos:
                for i in range(len(pos_row)):
                    mat_out[pos_row[i],pos_col[i],js] = mat_dict[str(ind+1)]
        
        return mat_out
    

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
    


    def run_ctqmc(self):
        
        #run_cmd = 'mpirun -np 1 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/CTQMC params'
        print(run_cmd)
        
        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def measure_ctqmc(self):
        
        # run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/EVALSIM params'
        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/EVALSIM params'

        print(run_cmd)
        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd,shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        print("measure self-energy done")

        return None
    

    def impurity_postprocessing(self, key,iter,equiv): # key -> problem number
    
        print("*****************************")
        print("Impurity Postprocessing Strat")
        print("*****************************")
        print(f'key : {key}')
        fileobs='./params.obs.json'
        filemeas='./params.meas.json'
        
        obsjson = json.load(open(fileobs))
        obsjson = obsjson['partition']

        
    
        histo_temp=obsjson["expansion histogram"]
    
        histo=np.zeros((np.shape(histo_temp)[0], 2))
        histo[:,0]=np.arange(np.shape(histo_temp)[0])
        histo[:,1]=histo_temp

        nn=obsjson["scalar"]["N"]       
        ctqmc_sign=obsjson["sign"]
    
        # histogram
        firstmoment=sum(histo[:,0]*histo[:,1])/sum(histo[:,1])
        secondmoment=sum((histo[:,0]-firstmoment)**2*histo[:,1])/sum(histo[:,1])

        print('first moment',  firstmoment)
        print('second moment', secondmoment)

        green = {}
        for key,val in obsjson["green"].items():
            templist = []
            for ii in range(len(val['function']['real'])):
                templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
            green[key]=templist
        Green = self.read_dict_LocDyn(equiv,green)
        sigma_bare = {}
        sigma_hf = {}
        for key, val in obsjson["self-energy"].items():
            sigma_hf[key] = complex(val['moments'][0])
            templist = []
            for ii in range(len(val['function']['real'])):
                templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j)
            sigma_bare[key] = templist
        Sigma_hf = self.read_dict_LocStc(equiv,sigma_hf)
        Sigma_bare = self.read_dict_LocDyn(equiv,sigma_bare)
        # Sigma_bare = self.FmixLocDyn(iter,0.05,Sigma_bare,self.Sigma_temp)
     
        params = json.load(open('./params.json'))
        cutoff = params["partition"]["green matsubara cutoff"]
        
        Sigma_bare = self.FgaussianLocDyn(self.ft.omega,Sigma_bare,0.05,1/self.ft.beta,cutoff)
    
        Green = self.FgaussianLocDyn(self.ft.omega,Green,0.05,1/self.ft.beta,cutoff)



        ndim = 6
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspin = 2

        print(norbc,ns)

        nft = len(obsjson["occupation-susceptibility-bulla"]['0_0']['function'])

        tempmat = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nft), dtype=np.complex128, order='F')

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb, ispin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb, jspin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind2,nn2)
                name = str(ind1)+'_'+str(ind2)
                
                if ispin<ns and jspin<ns:
                    # print(iorb,jorb,jorb,iorb,ispin,jspin)
                    tempmat[iorb,jorb,jorb,iorb,ispin,jspin,:] = obsjson["occupation-susceptibility-bulla"][name]["function"]


        susceptibility = np.copy(tempmat)

        
        # susceptibilit_temp = []
        # for i in range(len(obsjson["occupation-susceptibility-bulla"])):
        #     susceptibilit_temp.append([])
        #     for key,val in obsjson["occupation-susceptibility-bulla"].items():
        #         for ii in range(len(val['function'])):
        #             susceptibilit_temp[i].append(val['function'][ii])
                
        # norb = np.sqrt(len(obsjson["occupation-susceptibility-bulla"]))

        # susceptibilit = np.zeros((norb,norb,))

        print("******************************")
        print("Impurity Postprocessing Finish")
        print("******************************")
        return Green, Sigma_bare, Sigma_hf, susceptibility












    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = matin.shape[3]

        if os.path.exists('flocdyn'):
            pass
        else:
            os.mkdir("flocdyn")
        os.chdir("flocdyn")
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, ift, Re(F(w)), Im(F(w))\n")
            for ift in range(nft):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {ift} {matin[iorb,jorb,js,ift].real} {matin[iorb,jorb,js,ift].imag}\n")
        os.chdir("..")
        return None

class GreenLoc(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, green : GreenInt):

        super().__init__(crystal, ft)
        self.green = green
        self.gf = None
        self.gt = None

        self.occ = None

        self.Cal()

    def Cal(self): # projection

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        # nft = self.ft.size
        nft = len(self.ft.omega) ### ??
        ntau = len(self.ft.tau)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        # print(self.crystal.fprojector.shape[1])
        # print(self.crystal.ns)
        # print(len(self.ft.tau))
        # print(self.crystal.fprojector.shape[3])
        # print(self.crystal.fprojector.shape)

        # print('====')


        gf = np.zeros((norbc,norbc,ns,nft,nprob),dtype=np.complex128)
        self.gt = np.zeros((norbc,norbc,ns,ntau,nprob),dtype=np.complex128)

        # print(self.green.kf.shape)
        # print(self.green.kf[0,0,0,10,5])
        # print(self.green.kf[0,1,0,10,5])

        # exit()
        gf = self.green.Projection(self.green.kf)
        # for ispace in range(nspace):
        #     gf[...,ispace] = QAFort.projection.flatdyn(self.green.kf,self.crystal.fprojector[...,ispace])
        # for iprob in range(nprob):
            # gf[...,iprob] = QAFort.projection.flatdyn(self.green.kf,self.crystal.fprojector[...,iprob])

        self.gf = self.Loc2Imp(gf)
        ### conversion Loc -> Imp

        # gff = self.Loc2Imp(gf)
        # gff = gf
        # self.gf = gff
        for iprob in range(nprob):
            self.gt[...,iprob] = self.F2T(self.gf[...,iprob],1,1) ### ?

        return None

    def Occ(self):

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspace = self.crystal.fprojector.shape[3]
        # nrk = len(self.crystal.kpoint)
        nprob = len(self.crystal.probspace)


        # occk = np.zeros((norbc,norbc,ns,nrk),dtype=np.complex128,order='F')
        occ = np.zeros((norbc,norbc,ns,nprob),dtype=np.complex128,order='F')

        # print("Density matrixy calculation start")

        occ = -self.gt[...,-1,:]

        # occk = -self.rt[...,-1]

        # for irk in range(nrk):
        #     occ += occk[...,irk]

        # occ /= nrk
        self.occ = occ
        # self.occk = occk

        # self.occr = self.flatstc.K2R(occk)
        # print("Density matrixy calculation finish")
        return None

    def NumofE(self):

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        # nrk = len(self.crystal.kpoint)
        nft = len(self.ft.omega)#self.ft.size
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        # tempmat = copy.deepcopy(self.gkfmu0)
        # chem = self.ChemEmbedding(mu)
        # gcalf = np.zeros((norbc,norbc,ns,nft),dtype=np.complex128,order='F')
        # gcalt = np.zeros((norbc,norbc,ns,nft),dtype=np.complex128,order='F')


        # gcalf = self.Dyson(tempmat,-chem)
        # gcalt = self.F2T(gcalf,1,1)

        Ne = 0

        # for irk in range(nrk):
        for iprob in range(nprob):
            for js in range(ns):
                for iorb in range(norbc):
                    Ne += -np.real(self.gt[iorb,iorb,js,-1,iprob])

        # Ne /= nrk

        N = self.crystal.nume
        # print(N,Ne,N-Ne)

        return N - Ne

    # def densitymatrix()

class GreenImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLoc(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, sigma : object):
        super().__init__(crystal, ft)

        self.sigma = sigma
        self.f = None
        self.Cal()

    def Cal(self): # projection

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = self.ft.size
        nspace = self.crystal.fprojector.shape[3]

        sigmalocf = np.zeros((norbc,norbc,ns,nft,nspace),dtype=np.complex128,order='F')

        for isapce in range(nspace):
            sigmalocf[...,isapce] = QAFort.projection.flatdyn(self.sigma,self.crystal.fprojector[...,isapce])

        self.f = self.Loc2Imp(sigmalocf)

        # self.f = sigmalocf
        self.t = self.F2T(sigmalocf,0,1)

        return None


class SigmaImp(FLocDyn): # read CTQMC output

    def __init__(self, crystal: Crystal, ft: FTGrid):
        super().__init__(crystal, ft)
        self.Cal()

    def Cal(self):
        super().Dict2Arr()
        pass

class SigmaLGWC(FLocDyn):
# class SigmaLGWC(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid
    ,green : np.ndarray = None, wloc : np.ndarray = None, hdf5file : str = 'glob.h5',group : str = None):
        super().__init__(crystal, ft)

        # pass

        self.flpcstc = FLocStc(crystal=crystal)
        self.rt = None
        self.rf = None
        # self.kt = None
        # self.kf = None
        self.stck = None
        self.z = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        if green is None:
            print("Error, green doesn't exist")
            sys.exit()

        if wloc is None:
            print("Error, wloc doesn't exist")
            sys.exit()
        self.green = green
        self.wloc = wloc
        self.Cal()

    def Cal(self)->np.ndarray: #SigmaGWC
        '''
        Generate correlated self-energy
        input : Wc(R,t), G(R,t)

        return : crtau, crfreq, cktau, ckfreq
        '''

        G = self.green
        Wc = self.wloc
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        # nr = G.shape[3]
        ntau = len(self.ft.tau)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)
        norb = self.crystal.bprojector.shape[1]

        crtau = np.zeros((norbc,norbc,ns,ntau,nprob),dtype=np.complex128,order='F')

        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128,order='F')

        for itau in range(ntau):
            for iprob in range(nprob):
                for ind2 in range(norb*ns):
                    nn2 = [0]*2
                    ind2, [jorb,ks] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind2,nn2)
                    [b,[m3,m2]] = self.crystal.BAtomOrb(jorb)
                    iorbc3 = self.crystal.FIndex([b,m3])
                    iorbc2 = self.crystal.FIndex([b,m2])
                    for ind1 in range(norb*ns):
                        nn1 = [0]*2
                        ind1, [iorb,js] = self.crystal.indexing(norb*ns,2,[norb,ns],0,ind1,nn1)
                        [a,[m1,m4]] = self.crystal.BAtomOrb(iorb)
                        iorbc1 = self.crystal.FIndex([a,m1])
                        iorbc4 = self.crystal.FIndex([a,m4])
                        if js==ks:
                            crtau[iorbc1,iorbc2,js,itau,iprob] += -G[iorbc4,iorbc3,js,itau,iprob]*Wc[iorb,jorb,js,ks,itau,iprob]



        # cktau = self.R2K(crtau)
        # ckfreq = self.T2F(cktau)
        crfreq = np.zeros((norbc,norbc,ns,len(self.ft.omega),nprob),dtype=np.complex128,order='F')
        for iprob in range(nprob):
            crfreq[...,iprob] = self.T2F(crtau[...,iprob])

        self.rt = crtau
        # self.kt = cktau
        self.rf = crfreq
        # self.kf = ckfreq

        return None










class Hybridisation(FLocDyn):

    def __init__(self, crystal: Crystal, ft: FTGrid, gloc : GreenLoc, eimp : EImp, sigmahimp : np.array = None, sigmafimp : np.array = None, sigmacimp : np.array = None):
        super().__init__(crystal, ft)

        self.rt = None
        self.rf = None

        self.gfinv = None

        self.gloc = gloc
        self.eimp = eimp

        self.sigmahimp = sigmahimp
        self.sigmafimp = sigmafimp
        self.sigmacimp = sigmacimp

        self.nft = len(self.ft.omega)

        self.Cal()

    def Cal(self):

        import cmath

        imag_one = 1j

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nft = len(self.ft.omega)
        ntau = len(self.ft.tau)
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        rf = np.zeros((norbc,norbc,ns,nft,nprob),dtype=np.complex128,order='F')

        IdentityMatrix = np.identity(norbc)

        if self.sigmacimp is None:
            self.sigmacimp = np.zeros((norbc,norbc,ns,nft,nprob),dtype=np.complex128,order='F')

        gfinv = self.Inverse(self.gloc.gf)
        # self.gfinv = gfinv
        
        for iprob in range(nprob):
            for ift in range(nft):
                for js in range(ns):
                    rf[...,js,ift,iprob] = imag_one*self.ft.omega[ift]*IdentityMatrix[...] - gfinv[...,js,ift,iprob] - self.eimp.r[...,js,iprob] - (self.sigmahimp[...,js,iprob] + self.sigmafimp[...,js,iprob] + self.sigmacimp[...,js,ift,iprob])
        self.rf = rf

        self.rt = np.zeros((norbc,norbc,ns,ntau,nprob),dtype=np.complex128)
        for iprob in range(nprob):
            self.rt[...,iprob] = self.F2T(self.rf[...,iprob],0,1)
        

        return None
    

    # hyb_dict = self.fermion.write_dict_LocDyn(equiv_mat,self.fermion.hyb)
    # self.write_hyb_json(iter,str(iprob+1),hyb_dict)
    

    

    def write_hyb_json(self,iter,key,hyb : dict):

        if self.crystal.soc is False:
            if self.crystal.ns == 1:
                json_dict = {}
                for ikey,val in hyb.items():
                    json_dict[ikey] = {}
                    json_dict[ikey]['beta'] = self.ft.beta
                    json_dict[ikey]['real'] = np.real(val[0]).tolist()
                    json_dict[ikey]['imag'] = np.imag(val[0]).tolist()

                with open(f'hyb.{iter}.{key}.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
                with open('hyb.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC must be False")
            sys.exit()
        return None
    





    def write_ctqmc_params(self, iter, key, E_imp : EImp, equiv : np.ndarray, vloc : np.ndarray):
        
        if self.crystal.soc is False:
            if self.crystal.ns ==1:
                params = {}
                params["hloc"] = {}

                # mu_ctqmc=-np.real(E_imp[0,0,0])
                # # print(mu_ctqmc,type(mu_ctqmc))
                # E_imp = E_imp[:,:,0]+mu_ctqmc*np.eye(E_imp.shape[0],E_imp.shape[0])
                # E_imp = np.array(np.real(E_imp),dtype=float)
                # tempmat = np.kron(E_imp,np.eye(2,2))

                E_imp.A_final = np.array(np.real(E_imp.A_final),dtype=float)
                E_imp.ctqmc_mu = np.array(np.real(E_imp.ctqmc_mu),dtype=float)

                params["hloc"]['one body'] = np.round(E_imp.A_final[...,key-1]).tolist() ### forward backward !!!

                U = self.GetUijklComCTQMC(key, vloc)    ####### !!!
                params["hloc"]["two body"]=U.tolist() ####### !!!
                
                params["partition"]={}
                
                params["partition"]["green basis"]= "matsubara"
                params["partition"]["green bulla"]= True
                params["partition"]["green matsubara cutoff"] = self.ft.cutoff # 50
                params["partition"]["occupation susceptibility bulla"]=True
                params["partition"]["occupation susceptibility direct"]=False
                params["partition"]["quantum number susceptibility"] = True
                params["partition"]["susceptibility cutoff"]=self.ft.cutoff # 50
                params["partition"]["susceptibility tail"]=0 #200
                params["partition"]["quantum numbers"]={}
                tempmat = np.ones(E_imp.A_final[...,key-1].shape[0]*2)
                params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                for ii in range(len(tempmat)):
                    if ii < E_imp.A_final[...,key-1].shape[0]:
                        tempmat[ii]*= 0.5
                    elif ii >= E_imp.A_final[...,key-1].shape[0]:
                        tempmat[ii]*=-0.5
                params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                params["partition"]["probabilities"]={}
                params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                params["partition"]["density matrix precise"] = True
                params["partition"]["print eigenstates"] = True
                params["partition"]["print density matrix"]= True
                

                params["beta"]=self.ft.beta
                params["complex"] = False
                params["mu"]=E_imp.ctqmc_mu[key-1].tolist()
                params["hybridisation"]={}

                tempmat2 = np.kron(np.eye(2),equiv)  ###### equiv ??
                # tempmat2 = np.kron(equiv,np.eye(2))  ###### equiv ??
                tempmat2 = tempmat2.tolist()
                for ii in range(len(tempmat2)):
                    for jj in range(len(tempmat2)):
                        if tempmat2[ii][jj]==0.0:
                            tempmat2[ii][jj] = ""
                        else:
                            tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

                params["hybridisation"]["matrix"]=tempmat2
                params["hybridisation"]["functions"]="hyb.json"
                params["thermalisation time"]=1 #imp['thermalization_time']
                params["quantum number susceptibility"]=True
                params["occupation susceptibility bulla"]=True        
                params["green bulla"]=True       
                params["density matrix precise"]=False #True 
                params["measurement time"]=3 #imp['measurement_time']
                
                with open(f'params.{iter}.{key}.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                with open('params.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                # print("params.json written", file=self.m_ini.control['h_log'])
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC is not  False, please change SOC")
            sys.exit()

        return None
    




    def GetUijklComCTQMC(self, key, VLOC):

        norb = len(self.crystal.find)
        ns = self.crystal.ns

        # print(self.crystal.fimpdict)
        
        orb = self.crystal.fimpdict[str(key)][0]
        norbc = len(orb)
        # print(norbc)
        tempmat = np.zeros((norb, norb, norb, norb), dtype=np.complex128, order='F')
        vloc_temp = np.zeros((norbc, norbc, norbc, norbc, ns, ns), dtype=np.complex128, order='F')
        for ks in range(ns):
            for js in range(ns):
                tempmat = self.crystal.Double2Quad(VLOC[...,js,ks])
                # print(tempmat.shape)
                for ii, iorb in enumerate(orb):
                    for jj, jorb in enumerate(orb):
                        for kk, korb in enumerate(orb):
                            for ll, lorb in enumerate(orb):
                                # print(ii,jj,kk,ll,iorb,jorb,korb,lorb)
                                vloc_temp[ii, jj, kk, ll, js, ks] = tempmat[iorb, jorb, korb, lorb]

        if (self.crystal.soc == False):
            U = np.zeros((norbc**4*2**4), dtype=np.float64, order='F')
            idx = 0
            if (ns == 1):
                for sl in range(2):
                    for l in range(norbc):
                        for sk in range(2):
                            for k in range(norbc):
                                for sj in range(2):
                                    for j in range(norbc):
                                        for si in range(2):
                                            for i in range(norbc):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val = vloc_temp[i, j, k, l, 0, 0].real
                                                    val = abs(val)
                                                    if (val > 0.001):
                                                        U[idx] = val
                                                idx += 1
            else:
                for sl in range(2):
                    for l in range(norbc):
                        for sk in range(2):
                            for k in range(norbc):
                                for sj in range(2):
                                    for j in range(norbc):
                                        for si in range(2):
                                            for i in range(norbc):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val = vloc_temp[i, j, k, l, si, sj].real
                                                    val = abs(val)
                                                    if (val > 0.001):
                                                        U[idx] = val
                                                idx += 1
        else:
            print("SOC is not False")
            sys.exit()
        # self.u_ctqmc = U

        return U
    





class Weiss_Green(FLocDyn):
    def __init__(self, crystal: Crystal, ft: FTGrid, 
                 niham : NIHamiltonian, mu, hamh : SigmaHartree, hamf : SigmaFock, hloc : SigmaHLoc, floc : SigmaFLoc, 
                 gloc : GreenLoc, sigmahimp : np.array, sigmafimp : np.array, sigmacimp : np.array):
        super().__init__(crystal, ft)

        ### Eimp ###
        self.niham = niham
        self.mu = mu
        self.hamh = hamh
        self.hamf = hamf
        self.hloc = hloc
        self.floc = floc

        self.hamhf = self.niham.k + self.hamh.k + self.hamf.k
        self.sigmahfloc = self.hloc.r + self.floc.r

        self.Eimp_r = None
        self.Eimp_A_final = None
        self.Eimp_ctqmc_mu = None


        ### Delta ###
        self.delta_rt = None
        self.delta_rf = None

        self.gfinv = None

        self.gloc = gloc
        # self.eimp = eimp

        self.sigmahimp = sigmahimp
        self.sigmafimp = sigmafimp
        self.sigmacimp = sigmacimp

        self.Cal()

    def Cal(self):

        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        norb = self.hamhf.shape[0]
        nk = self.hamhf.shape[3]
        nspace = self.crystal.fprojector.shape[3]
        nprob = len(self.crystal.probspace)

        import cmath

        imag_one = 1j

        # norbc = self.crystal.fprojector.shape[1]
        # ns = self.crystal.ns
        nft = len(self.ft.omega)
        ntau = len(self.ft.tau)
        # nspace = self.crystal.fprojector.shape[3]
        # nprob = len(self.crystal.probspace)



        ### Eimp ###
        self.hamhf_projection = np.zeros_like(self.sigmahfloc)

        tempmat = np.zeros((norbc, norbc, ns, nspace), dtype=np.complex128, order='F')
        hamhf = np.copy(self.hamhf)
        
        for ik in range(nk):
            for js in range(ns):
                for iorb in range(norb):
                    hamhf[iorb, iorb, js, ik] -= self.mu
        
        for ispace in range(nspace):
            tempmat[...,ispace] = QAFort.projection.flatstc(hamhf,self.crystal.fprojector[...,ispace])
        
        self.hamhf_projection = self.Loc2Imp_Stc(tempmat)

        self.Eimp_r = np.zeros((norbc,norbc,ns,nprob), dtype=np.complex128, order='F')
        for iis in range(ns):
            for iprob in range(nprob):
                self.Eimp_r[...,iis,iprob] = self.hamhf_projection[...,iis,iprob] - self.sigmahfloc[...,iis,iprob]
    



        ### Delta ###
        delta_rf = np.zeros((norbc,norbc,ns,nft,nprob),dtype=np.complex128,order='F')

        IdentityMatrix = np.identity(norbc)

        if self.sigmacimp is None:
            self.sigmacimp = np.zeros((norbc,norbc,ns,nft,nprob),dtype=np.complex128,order='F')

        gfinv = self.Inverse(self.gloc.gf)
        # self.gfinv = gfinv
        
        for iprob in range(nprob):
            for ift in range(nft):
                for js in range(ns):
                    delta_rf[...,js,ift,iprob] = imag_one*self.ft.omega[ift]*IdentityMatrix[...] - gfinv[...,js,ift,iprob] - self.Eimp_r[...,js,iprob] - (self.sigmahimp[...,js,iprob] + self.sigmafimp[...,js,iprob] + self.sigmacimp[...,js,ift,iprob])
        self.delta_rf = delta_rf

        self.delta_rt = np.zeros((norbc,norbc,ns,ntau,nprob),dtype=np.complex128)
        for iprob in range(nprob):
            self.delta_rt[...,iprob] = self.F2T(self.delta_rf[...,iprob],0,1)





    def Eimp_final_input(self): ## move to EImp

        nprob = len(self.crystal.probspace)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]

        if ns==1:
            mu = np.zeros(nprob, dtype=np.complex128, order='F')
            # for i in range(nprob):
            #     mu[i] = -B[0,0,i]  ### is the mu the same along the omega space?
            I = np.identity(len(self.Eimp_r))
            A = np.zeros((norbc,norbc), dtype=np.complex128, order='F')
            A_final = np.zeros((norbc*2,norbc*2,nprob), dtype=np.complex128, order='F')

            for i in range(nprob):
                mu[i] = -self.Eimp_r[0,0,0,i]
                A = self.Eimp_r[...,0,i] + mu[i]*I
                A_final[...,i] = np.kron(np.eye(2),A)

            self.Eimp_A_final = np.copy(A_final)
            self.Eimp_ctqmc_mu = np.copy(mu)
        
        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
        
        return None






class Sigma_imp(FLocDyn):
    def __init__(self, crystal: Crystal, ft: FTGrid): 
        super().__init__(crystal, ft)



    def write_hyb_json(self,iter,key,hyb : dict):

        if self.crystal.soc is False:
            if self.crystal.ns == 1:
                json_dict = {}
                for ikey,val in hyb.items():
                    json_dict[ikey] = {}
                    json_dict[ikey]['beta'] = self.ft.beta
                    json_dict[ikey]['real'] = np.real(val[0]).tolist()
                    json_dict[ikey]['imag'] = np.imag(val[0]).tolist()

                with open(f'hyb.{iter}.{key}.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
                with open('hyb.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC must be False")
            sys.exit()
        return None
    

    def write_ctqmc_params(self, iter, key, weiss_green : Weiss_Green, equiv : np.ndarray, vloc : np.ndarray):
        
        if self.crystal.soc is False:
            if self.crystal.ns ==1:
                params = {}
                params["hloc"] = {}

                # mu_ctqmc=-np.real(E_imp[0,0,0])
                # # print(mu_ctqmc,type(mu_ctqmc))
                # E_imp = E_imp[:,:,0]+mu_ctqmc*np.eye(E_imp.shape[0],E_imp.shape[0])
                # E_imp = np.array(np.real(E_imp),dtype=float)
                # tempmat = np.kron(E_imp,np.eye(2,2))

                weiss_green.Eimp_A_final = np.array(np.real(weiss_green.Eimp_A_final),dtype=float)
                weiss_green.Eimp_ctqmc_mu = np.array(np.real(weiss_green.Eimp_ctqmc_mu),dtype=float)

                params["hloc"]['one body'] = weiss_green.Eimp_A_final[...,key-1].tolist() ### forward backward !!!

                U = self.GetUijklComCTQMC(key, vloc)    ####### !!!
                params["hloc"]["two body"]=U.tolist() ####### !!!
                
                params["partition"]={}
                
                params["partition"]["green basis"]= "matsubara"
                params["partition"]["green bulla"]= True
                params["partition"]["green matsubara cutoff"] = self.ft.cutoff # 50
                params["partition"]["occupation susceptibility bulla"]=True
                params["partition"]["occupation susceptibility direct"]=False
                params["partition"]["quantum number susceptibility"] = True
                params["partition"]["susceptibility cutoff"]=self.ft.cutoff # 50
                params["partition"]["susceptibility tail"]=200
                params["partition"]["quantum numbers"]={}
                tempmat = np.ones(weiss_green.Eimp_A_final[...,key-1].shape[0]*2)
                params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                for ii in range(len(tempmat)):
                    if ii < weiss_green.Eimp_A_final[...,key-1].shape[0]:
                        tempmat[ii]*= 0.5
                    elif ii >= weiss_green.Eimp_A_final[...,key-1].shape[0]:
                        tempmat[ii]*=-0.5
                params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                params["partition"]["probabilities"]={}
                params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                params["partition"]["density matrix precise"] = True
                params["partition"]["print eigenstates"] = True
                params["partition"]["print density matrix"]= True
                

                params["beta"]=self.ft.beta
                params["complex"] = False
                params["mu"]=weiss_green.Eimp_ctqmc_mu[key-1].tolist()
                params["hybridisation"]={}

                tempmat2 = np.kron(np.eye(2),equiv)  ###### equiv ??
                # tempmat2 = np.kron(equiv,np.eye(2))  ###### equiv ??
                tempmat2 = tempmat2.tolist()
                for ii in range(len(tempmat2)):
                    for jj in range(len(tempmat2)):
                        if tempmat2[ii][jj]==0.0:
                            tempmat2[ii][jj] = ""
                        else:
                            tempmat2[ii][jj] = str(int(tempmat2[ii][jj]))

                params["hybridisation"]["matrix"]=tempmat2
                params["hybridisation"]["functions"]="hyb.json"
                params["thermalisation time"]=1 #imp['thermalization_time']
                params["quantum number susceptibility"]=True
                params["occupation susceptibility bulla"]=True        
                params["green bulla"]=True       
                params["density matrix precise"]=False #True 
                params["measurement time"]=3 #imp['measurement_time']
                
                with open(f'params.{iter}.{key}.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                with open('params.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                # print("params.json written", file=self.m_ini.control['h_log'])
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC is not  False, please change SOC")
            sys.exit()

        return None
    

    def GetUijklComCTQMC(self, key, VLOC):

        norb = len(self.crystal.find)
        ns = self.crystal.ns

        # print(self.crystal.fimpdict)
        
        orb = self.crystal.fimpdict[str(key)][0]
        norbc = len(orb)
        # print(norbc)
        tempmat = np.zeros((norb, norb, norb, norb), dtype=np.complex128, order='F')
        vloc_temp = np.zeros((norbc, norbc, norbc, norbc, ns, ns), dtype=np.complex128, order='F')
        for ks in range(ns):
            for js in range(ns):
                tempmat = self.crystal.Double2Quad(VLOC[...,js,ks])
                # print(tempmat.shape)
                for ii, iorb in enumerate(orb):
                    for jj, jorb in enumerate(orb):
                        for kk, korb in enumerate(orb):
                            for ll, lorb in enumerate(orb):
                                # print(ii,jj,kk,ll,iorb,jorb,korb,lorb)
                                vloc_temp[ii, jj, kk, ll, js, ks] = tempmat[iorb, jorb, korb, lorb]

        if (self.crystal.soc == False):
            U = np.zeros((norbc**4*2**4), dtype=np.float64, order='F')
            idx = 0
            if (ns == 1):
                for sl in range(2):
                    for l in range(norbc):
                        for sk in range(2):
                            for k in range(norbc):
                                for sj in range(2):
                                    for j in range(norbc):
                                        for si in range(2):
                                            for i in range(norbc):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val = vloc_temp[i, j, k, l, 0, 0].real
                                                    val = abs(val)
                                                    if (val > 0.001):
                                                        U[idx] = val
                                                idx += 1
            else:
                for sl in range(2):
                    for l in range(norbc):
                        for sk in range(2):
                            for k in range(norbc):
                                for sj in range(2):
                                    for j in range(norbc):
                                        for si in range(2):
                                            for i in range(norbc):
                                                    
                                                    
                                                if(sj==sk and si==sl):
                                                    val = vloc_temp[i, j, k, l, si, sj].real
                                                    val = abs(val)
                                                    if (val > 0.001):
                                                        U[idx] = val
                                                idx += 1
        else:
            print("SOC is not False")
            sys.exit()
        # self.u_ctqmc = U

        return U