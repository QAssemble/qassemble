import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLocDyn import *
from .FLocStc import *
from .BLocDyn import *
from .BLocStc import *

class CTQMC(object):

    def __init__(self):
        
        
        pass

    def PreProcessing(self, iter : int, key : int, **kwargs):

        Eimp = kwargs['Eimp']
        imp = kwargs['imp']
        equiv = kwargs['equiv']
        vloc = kwargs['vloc']
        hyb = kwargs['hyb']
        bweiss = kwargs['bweiss']

        # Use class directly
        hyb_dict = Hybridisation().write_hyb_dict() 

        Hybridisation().write_json()
        

        return None
    
    def Run(self):

        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/CTQMC params'

        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def Measure(self):
        
        # run_cmd = 'mpirun -np 4 '+diage_path+'/ComCTQMC/bin/EVALSIM params'
        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/EVALSIM params'

        
        with open('./evalsim.out', 'w') as logfile, open('./evalsim.err', 'w') as errfile :
            ret = subprocess.call(run_cmd,shell=True, stdout=logfile, stderr=errfile)
            if ret != 0:
                print("Error in EVALSIM. Check evalsim.err for error message.")
                sys.exit()
        print("measure self-energy done")

        return None
    
    def PostProcessing(self,iter,key,**kwargs): 

        equiv = kwargs['equiv']
        utilde_rf = kwargs['utilde_rf']
        
        pass

    def imp_B2F(self,imp,B,key):

        _,_,ns=B.shape
        if ns==1:

            self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index

            nprob = len(self.crystal.probspace)

            F = {}
            
            # for ii in range(nprob):

            #     iimp = str(ii+1)

            # F[iimp] = {}
            for i in range(len(self.crystal.imp_index[key-1])):
                index_of_equivalance = str(i+1)
                F[index_of_equivalance] = 0

                for j in range(len(self.crystal.imp_index[key-1][i])):
                    F[index_of_equivalance] = F[index_of_equivalance] + B[self.crystal.imp_index[key-1][i][j][0],self.crystal.imp_index[key-1][i][j][1],0]

                F[index_of_equivalance] = F[index_of_equivalance]/len(self.crystal.imp_index[key-1][i]) ### take the average
        
        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
            
        return F
    

    def imp_F2B(self,imp,F,key):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index
        
        nprob = len(self.crystal.probspace)
        norbc = self.crystal.fprojector.shape[1]

        B = np.zeros((norbc,norbc), dtype=np.complex128, order='F')
        # ii = 0 # index of impurity problems -- nprob
        # for key,val in F.items():
        i=0 # index of equivalances
        for valkey,valval in F.items():
            for j in range(len(self.crystal.imp_index[key-1][i])):
                B[self.crystal.imp_index[key-1][i][j][0],self.crystal.imp_index[key-1][i][j][1]] = valval
            i += 1
            # ii += 1
        
        return B
    
    def Eimp_final_input(self, Eimp): ## move to EImp

        # nprob = len(self.crystal.probspace)
        ns = self.crystal.ns
        norbc = self.crystal.fprojector.shape[1]

        if ns==1:
            # ctqmc_mu = np.zeros(nprob, dtype=np.complex128, order='F')
            # for i in range(nprob):
            #     mu[i] = -B[0,0,i]  ### is the mu the same along the omega space?
            I = np.identity(len(Eimp))
            A = np.zeros((norbc,norbc), dtype=np.complex128, order='F')
            A_final = np.zeros((norbc*2,norbc*2), dtype=np.complex128, order='F')

            # for i in range(nprob):
            ctqmc_mu = -Eimp[0,0]
            A = Eimp[...,0] + ctqmc_mu*I
            A_final[...] = np.kron(np.eye(2),A)

            # self.A_final = np.copy(A_final)
            # self.ctqmc_mu = np.copy(mu)
        
        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
        
        return A_final,ctqmc_mu
    
    def imp_B2F_freq(self,imp,B,key):

        _,_,ns,nft=B.shape
        if ns==1:

            self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index

            nprob = len(self.crystal.probspace)
            # nft = len(self.ft.omega)

            F = {}
            
            # for ii in range(nprob):

            #     iimp = str(ii+1)

            # F[iimp] = {}
            for i in range(len(self.crystal.imp_index[key-1])):
                index_of_equivalance = str(i+1)
                
                for k in range(nft):
                    try:
                        # F[iimp][index_of_equivalance].append(0)
                        F[index_of_equivalance].append(0)
                    except KeyError:
                        # F[iimp][index_of_equivalance] = [0]
                        F[index_of_equivalance] = [0]

                for k in range(nft):
                    for j in range(len(self.crystal.imp_index[key-1][i])):
                        F[index_of_equivalance][k] = F[index_of_equivalance][k] + B[self.crystal.imp_index[key-1][i][j][0],self.crystal.imp_index[key-1][i][j][1],0,k]

                    F[index_of_equivalance][k] = F[index_of_equivalance][k]/len(self.crystal.imp_index[key-1][i]) ### take the average

        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
            
        return F


    def imp_F2B_freq(self,imp,F,key):

        self.crystal.read_imp_equi_mat(imp)  ## read imp_equivalant_mat and store information in crystal.imp_index
        
        # print(len(F['1']))

        print(np.array(self.crystal.imp_index).shape)

        # exit()

        nprob = len(self.crystal.probspace)
        nft = len(self.ft.omega)
        norbc = self.crystal.fprojector.shape[1]

        B = np.zeros((norbc,norbc,nft),dtype=np.complex128,order='F')

        # ii = 0 # index of impurity problems -- nprob
        # for key,val in F.items():
        i=0 # index of equivalances
        for valkey,valval in F.items():
            for j in range(len(self.crystal.imp_index[key-1][i])):
                for k in range(nft): # number of omega -- nft
                    print(len(valval),nft,i,j,k,self.crystal.imp_index[key-1][i][j][0],self.crystal.imp_index[key-1][i][j][1],k)
                    B[self.crystal.imp_index[key-1][i][j][0],self.crystal.imp_index[key-1][i][j][1],k] = valval[k]
                i += 1
            # ii += 1

        return B
    
