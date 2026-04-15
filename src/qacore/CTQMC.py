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

    def __init__(self, crystal : Crystal, ft : FTGrid):

        self.crystal = crystal
        self.ft = ft
        
        
        # pass

    def PreProcessing(self, iter : int, key : int, **kwargs):

        # iter = iter + 1 ### convert index from 0-based to 1-based

        Eimp = kwargs['Eimp']
        imp = kwargs['imp']
        equiv = kwargs['equiv']
        vloc = kwargs['vloc']
        hyb = kwargs['hyb']
        bweiss = kwargs['bweiss']

        # # Use class directly
        # hyb_dict = Hybridisation().write_hyb_dict() 

        # Hybridisation().write_json()


        if equiv is None:
            equiv = self.crystal.read_imp_equi_mat(imp)

        flocdyn = FLocDyn(self.crystal,self.ft)
        blocdyn = BLocDyn(self.crystal,self.ft)

        ###########################
        ### Write hyb.json file ###
        ###########################
        print('*** write hyb.json file ***')
        # delta_F = self.imp_B2F_freq(imp,Hyb[...,key-1],key)  ### use this
        # delta_B = self.imp_F2B_freq(imp,delta_F,key)
        # hyb_dict = self.write_hyb_dict(equiv,hyb[...,key-1])
        # self.write_hyb_json(iter,key,hyb_dict)
        hyb_dict = flocdyn.write_hyb_dict(equiv,hyb[...,key-1])
        flocdyn.write_hyb_json(iter,key,hyb_dict)

        ###########################
        ### Write dyn.json file ###
        ###########################
        print('*** write dyn.json file ***')
        F0_dict = blocdyn.write_dyn_dict(1,1,bweiss)
        blocdyn.write_dyn_json(iter,key,F0_dict)

        ##############################
        ### Write params.json file ###
        ##############################
        if self.crystal.soc is False:
            if self.crystal.ns ==1:
                params = {}
                params["hloc"] = {}


                ### read impurity equivalent matrix and Eimp and separate into A_final and ctqmc_mu
                eimp_F = self.imp_B2F(imp,Eimp[...,key-1],key)
                eimp_B = self.imp_F2B(imp,eimp_F,key)
                Eimp_final,ctqmc_mu = self.Eimp_final_input(eimp_B)
                ### convert complex numbers to float numbers
                Eimp_final = np.array(np.real(Eimp_final),dtype=float)
                ctqmc_mu = np.array(np.real(ctqmc_mu),dtype=float)

                params["hloc"]['one body'] = np.round(Eimp_final).tolist()

                U = vloc.GetUijklComCTQMC(key)    ####### !!!
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
                # tempmat = np.ones(Eimp_final.shape[0]*2)
                tempmat = np.ones(Eimp_final.shape[0])
                params["partition"]["quantum numbers"]["N"]=tempmat.tolist()
                for ii in range(len(tempmat)):
                    if ii < Eimp_final.shape[0]:
                        tempmat[ii]*= 0.5
                    elif ii >= Eimp_final.shape[0]:
                        tempmat[ii]*=-0.5
                params["partition"]["quantum numbers"]["Sz"]=tempmat.tolist() # make 
                params["partition"]["probabilities"]={}
                params["partition"]["probabilities"]=["N","energy","Sz"]#["N","energy","S2","Sz"]
                params["partition"]["density matrix precise"] = True
                params["partition"]["print eigenstates"] = True
                params["partition"]["print density matrix"]= True
                

                params["beta"]=self.ft.beta
                params["complex"] = False
                params["mu"]=ctqmc_mu.tolist()
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
                params["measurement time"]=3 # 10 # 3 #imp['measurement_time']


                #### dyn.json ####
                params["dyn"] = {}
                params["dyn"]["matrix"] = [["F0"]]
                params["dyn"]["functions"] = "dyn.json"
                quantumnumber_temp = [[]]
                for i in range(len(Eimp_final[0])):
                    quantumnumber_temp[0].append(1)
                params["dyn"]["quantum numbers"] = quantumnumber_temp
                
                with open(f'params.{iter}.{key}.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                with open('params.json','w') as outfile:
                    json.dump(params,outfile, sort_keys=True, indent=4, separators=(',', ': '))
                    # json.dump(params,outfile, sort_keys=True, separators=(']'))
                # print("params.json written", file=self.m_ini.control['h_log'])
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC is not  False, please change SOC")
            sys.exit()
        

        return None
    
    def Run(self):

        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/CTQMC params'
        # run_cmd = 'mpirun -np '+str()+'~/DiagE/ComCTQMC/bin/CTQMC params'

        ## input nb of processors

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
        # print("measure self-energy done")

        return None
    
    def PostProcessing(self,iter,key,**kwargs): 

        iter = iter + 1 ### convert index from 0-based to 1-based

        equiv = kwargs['equiv']
        # utilde_rf = kwargs['utilde_rf']
        
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
                templist.append(val['function']['real'][ii]+val['function']['imag'][ii]*1j - sigma_hf[key])
            sigma_bare[key] = templist
        Sigma_hf = self.read_dict_LocStc(equiv,sigma_hf)
        Sigma_bare = self.read_dict_LocDyn(equiv,sigma_bare)
        # Sigma_bare = self.FmixLocDyn(iter,0.05,Sigma_bare,self.Sigma_temp)


        params = json.load(open('./params.json'))
        cutoff = params["partition"]["green matsubara cutoff"]

        flocdyn = FLocDyn(self.crystal,self.ft)
        Sigma_bare = flocdyn.GaussianLinearBroad(self.ft.omega,Sigma_bare,0.05,1/self.ft.beta,cutoff)
        Green = flocdyn.GaussianLinearBroad(self.ft.omega,Green,0.05,1/self.ft.beta,cutoff)

        print(Green.shape)
        print(Sigma_bare.shape)

        print(Sigma_hf[0,0,0])

        np.savetxt("Hybridisation_Green.txt", np.column_stack((Green[0,0,0,:].real, Green[0,0,0,:].imag)), fmt="%.6f")
        np.savetxt("Hybridisation_Sigma_bare.txt", np.column_stack((Sigma_bare[0,0,0,:].real, Sigma_bare[0,0,0,:].imag)), fmt="%.6f")

        
        # Sigma_bare = self.FgaussianLocDyn(self.ft.omega,Sigma_bare,0.05,1/self.ft.beta,cutoff)
        # Green = self.FgaussianLocDyn(self.ft.omega,Green,0.05,1/self.ft.beta,cutoff)

        susceptibility = self.read_susceptibility_LocDyn(equiv, obsjson)

        print(susceptibility.shape)

        np.savetxt("Hybridisation_susceptibility.txt", np.column_stack((susceptibility[0,0,0,0,0,0,:].real, susceptibility[0,0,0,0,0,0,:].imag)), fmt="%.6f")


        # exit()

        return Green,Sigma_hf,Sigma_bare,susceptibility




    def PostProcessing2(self,iter,key,**kwargs):

        Green = kwargs['Green']
        Sigma_hf = kwargs['Sigma_hf']
        Sigma_bare = kwargs['Sigma_bare']
        susceptibility = kwargs['susceptibility']
        utilde_rf = kwargs['utilde_rf']


        #############################################################
        #### separate Sigma_h and Sigma_f using Green's function ####
        #### & build classes for SigmaHImp, SigmaFImp, SigmaIGWC ####
        #############################################################
        ### build classes for Sigma_h, Sigma_f, Sigma_c and initialize them
        if int(key)-1 == 0:
            Sigma_h = SigmaHImp(self.crystal)
            Sigma_f = SigmaFImp(self.crystal)
            Sigma_c = SigmaCImp(self.crystal,self.ft)

        ### compute rho using Green's function
        flocdyn = FLocDyn(self.crystal,self.ft)
        Green_tau = flocdyn.F2T(Green,1,1)
        occ = (-1) * Green_tau[:, :, :, -1].copy()
        print('density - Occ')
        print(occ)   ### check density

        ### add Simga_h, Sigma_f and Sigma_c to each key (problem space) index
        Sigma_h.add_key(occ,utilde_rf,int(key)-1)         ## int(key)-1 convert 1-based to 0-based indexing
        Sigma_f.add_key(Sigma_hf,Sigma_h.r,int(key)-1)    ## Sigma_f = Sigma_hf - Sigma_h
        Sigma_c.add_key(Sigma_bare,int(key)-1)


        #############################################################
        #######    read susceptibility and compute Pi_emft    #######
        #############################################################
        ### read susceptibility
        # susceptibility = self.read_susceptibility_LocDyn(equiv, obsjson)

        ### compute Pi_edmft using susceptibility and utilde
        if int(key)-1 == 0:
            Pi = PolImp(self.crystal,self.ft)
        Pi.add_key(susceptibility, utilde_rf, int(key)-1)

        print("******************************")
        print("Impurity Postprocessing Finish")
        print("******************************")


        return Sigma_h, Sigma_f, Sigma_c, Pi   ## return classes - (SigmaHImp, SigmaFImp, SigmaIGWC, PolIGW)
    



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

            # print(Eimp.shape)
            # print(A_final.shape)

            # exit()

            # for i in range(nprob):

            # ctqmc_mu = -np.real(Eimp[0,0])
            # A = Eimp[...,0] + ctqmc_mu*I
            # A_final[...] = np.kron(np.eye(2),A)

            ctqmc_mu=-np.real(Eimp[0,0])
            A = Eimp[:,:]+ctqmc_mu*np.eye(Eimp.shape[0],Eimp.shape[0])
            A = np.array(np.real(A),dtype=float)
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
        
        # norb = len(equiv)
        # ns = self.crystal.ns
        # nfreq = len(mat_dict["1"])

        # mat_out = np.zeros((norb,norb,ns,nfreq),dtype=complex,order='F')

        # Nind = np.amax(equiv)
        
        # for js in range(ns):
        #     for ind in range(Nind):
        #         # pos = find_positions(equiv,ind+1) 
        #         pos_row, pos_col = np.where(equiv==ind+1)
        #         # for ii, jj in pos:
        #         for i in range(len(pos_row)):
        #             mat_out[pos_row[i],pos_col[i],js] = mat_dict[str(ind+1)]
        

        ndim = int(np.sqrt(len(mat_dict["occupation-susceptibility-bulla"])))
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspin = 2             ### nspin could be different from ns in CTQMC

        nft = len(mat_dict["occupation-susceptibility-bulla"]['0_0']['function'])

        mat_out = np.zeros((norbc,norbc,norbc,norbc,nspin,nspin,nft), dtype=np.complex128, order='F')

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb, ispin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb, jspin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind2,nn2)
                name = str(ind1)+'_'+str(ind2)
                mat_out[iorb,jorb,jorb,iorb,ispin,jspin,:] = mat_dict["occupation-susceptibility-bulla"][name]["function"]

        # susceptibility = np.copy(mat_out)
        
        return mat_out
    



    # def write_hyb_dict(self,equiv : np.ndarray, mat_in : np.ndarray)->dict:
        
    #     ns = mat_in.shape[2]
    #     Nind = int(np.amax(equiv))
    #     # print(Nind)
    #     # exit()
    #     mat_dict = {}    

    #     for ind in range(Nind):
    #         mat_dict[ind+1]=[]
    #         # pos = find_positions(equiv,ind+1)
    #         pos_row, pos_col = np.where(equiv==ind+1)
    #         for js in range(ns):
    #             e = 0
    #             # for ii, jj in pos:
    #             for i in range(len(pos_row)):
                    
    #                 e+=mat_in[pos_row[i],pos_col[i],js]
    #             e/=len(pos_row)
    #             mat_dict[ind+1].append(e.tolist())

    #     return mat_dict
    
    # def write_hyb_json(self,iter,key,hyb : dict):

    #     if self.crystal.soc is False:
    #         if self.crystal.ns == 1:
    #             json_dict = {}
    #             for ikey,val in hyb.items():
    #                 json_dict[ikey] = {}
    #                 json_dict[ikey]['beta'] = self.ft.beta
    #                 json_dict[ikey]['real'] = np.real(val[0]).tolist()
    #                 json_dict[ikey]['imag'] = np.imag(val[0]).tolist()

    #             with open(f'hyb.{iter}.{key}.json','w') as outfile:
    #                 json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
    #             with open('hyb.json','w') as outfile:
    #                 json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
    #                 # json.dump(json_dict,outfile,sort_keys=True, separators=(']'))
            
    #         elif self.crystal.ns == 2:
    #             print("Nspin is not 1")
    #             sys.exit()
    #     elif self.crystal.soc is True:
    #         print("SOC must be False")
    #         sys.exit()
    #     return None
    
    # def write_dyn_dict(self,iter,key,utilde_rf_2):
    #     norb,_,ns,_,nft,_ = utilde_rf_2.shape
    #     norbc = len(self.crystal.find)
    #     utilde_rf_4 = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nft),dtype=np.complex64,order='F')

    #     for iis in range(ns):
    #         for jjs in range(ns):
    #             for ift in range(nft):
    #                 utilde_rf_4[...,iis,jjs,ift] = self.crystal.Double2Quad(utilde_rf_2[...,iis,jjs,ift,0])
        
    #     F0_val = np.zeros(nft,dtype=np.float64, order='F')
    #     for ift in range(nft):
    #         F0_val[ift] = 1.0/ns**2/norbc**2*np.einsum('ijjimn->',utilde_rf_4[...,ift]).real
        
    #     F0_dict = {}
    #     F0_dict["F0"] = F0_val.tolist()

    #     return F0_dict
    

    # def write_dyn_json(self,iter,key,dyn : dict):

    #     if self.crystal.soc is False:
    #         if self.crystal.ns == 1:
    #             json_dict = dyn
    #             # for ikey,val in dyn.items():
    #             #     json_dict[ikey] = {}
    #             #     # json_dict[ikey]['beta'] = self.ft.beta
    #             #     json_dict[ikey]['real'] = np.real(val[0]).tolist()
    #             #     json_dict[ikey]['imag'] = np.imag(val[0]).tolist()

    #             # with open(f'hyb.{iter}.{key}.json','w') as outfile:
    #                 # json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
    #             with open('dyn.json','w') as outfile:
    #                 json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
    #         elif self.crystal.ns == 2:
    #             print("Nspin is not 1")
    #             sys.exit()
    #     elif self.crystal.soc is True:
    #         print("SOC must be False")
    #         sys.exit()
    #     return None
    
