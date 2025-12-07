import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *

class CorrelationFunction(object):

    def __init__(self, cry : dict = None, ft : dict = None, c = 1.0):

        self.c = c
        self.niham = None
        self.green = None
        self.greenbare = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.ham = None
        self.occ = None
        self.vbare = None
        self.pol = None
        self.w = None

        # cry = Crystal(latt=latt,basisposition=basisposition,ns=ns,soc=soc,rkgrid=rkgrid,orboption=orboption,N=N)
        #cry = Crystal#(Rvec=Rvec,CorF=CorF,Basis=Basis,Nspin=Nspin,SOC=SOC,Nelec=Nelec,#KGrid=KGrid)
        #self.cry = cry
        #ft = FTGrid(T=T,beta=beta,cutoff=cutoff)
        #self.ft = ft
        self.crystal = Crystal(cry=cry)
        self.ft = FTGrid(ft=ft)

        # if os.path.exists('work'):
        #     pass
        # else:
        #     os.mkdir('work')
    


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
    



    def write_hyb_dict(self,equiv : np.ndarray, mat_in : np.ndarray)->dict:
        
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
                    # json.dump(json_dict,outfile,sort_keys=True, separators=(']'))
            
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC must be False")
            sys.exit()
        return None
    

    def write_dyn_dict(self,iter,key,utilde_rf_2):
        norb,_,ns,_,nft,_ = utilde_rf_2.shape
        norbc = len(self.crystal.find)
        utilde_rf_4 = np.zeros((norbc,norbc,norbc,norbc,ns,ns,nft),dtype=np.complex64,order='F')

        for iis in range(ns):
            for jjs in range(ns):
                for ift in range(nft):
                    utilde_rf_4[...,iis,jjs,ift] = self.crystal.Double2Quad(utilde_rf_2[...,iis,jjs,ift,0])
        
        F0_val = np.zeros(nft,dtype=np.float64, order='F')
        for ift in range(nft):
            F0_val[ift] = 1.0/ns**2/norbc**2*np.einsum('ijjimn->',utilde_rf_4[...,ift]).real
        
        F0_dict = {}
        F0_dict["F0"] = F0_val.tolist()

        return F0_dict
    

    def write_dyn_json(self,iter,key,dyn : dict):

        if self.crystal.soc is False:
            if self.crystal.ns == 1:
                json_dict = dyn
                # for ikey,val in dyn.items():
                #     json_dict[ikey] = {}
                #     # json_dict[ikey]['beta'] = self.ft.beta
                #     json_dict[ikey]['real'] = np.real(val[0]).tolist()
                #     json_dict[ikey]['imag'] = np.imag(val[0]).tolist()

                # with open(f'hyb.{iter}.{key}.json','w') as outfile:
                    # json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
                with open('dyn.json','w') as outfile:
                    json.dump(json_dict,outfile,sort_keys=True, indent=4, separators=(',', ': '))
            
            elif self.crystal.ns == 2:
                print("Nspin is not 1")
                sys.exit()
        elif self.crystal.soc is True:
            print("SOC must be False")
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



    def CTQMCPreProcessing(self, iter, key, E_imp : np.ndarray, imp : dict, equiv : np.ndarray, vloc : np.ndarray, Hyb : np.ndarray, bweiss : np.ndarray):
        
        if equiv is None:
            equiv = self.crystal.read_imp_equi_mat(imp)

        ###########################
        ### Write hyb.json file ###
        ###########################
        print('*** write hyb.json file ***')
        # delta_F = self.imp_B2F_freq(imp,Hyb[...,key-1],key)  ### use this
        # delta_B = self.imp_F2B_freq(imp,delta_F,key)
        hyb_dict = self.write_hyb_dict(equiv,Hyb[...,key-1])
        self.write_hyb_json(iter,key,hyb_dict)

        ###########################
        ### Write dyn.json file ###
        ###########################
        print('*** write dyn.json file ***')
        F0_dict = self.write_dyn_dict(1,1,bweiss)
        self.write_dyn_json(iter,key,F0_dict)


        ##############################
        ### Write params.json file ###
        ##############################
        if self.crystal.soc is False:
            if self.crystal.ns ==1:
                params = {}
                params["hloc"] = {}


                ### read impurity equivalent matrix and E_imp and separate into A_final and ctqmc_mu
                eimp_F = self.imp_B2F(imp,E_imp[...,key-1],key)
                eimp_B = self.imp_F2B(imp,eimp_F,key)
                Eimp_final,ctqmc_mu = self.Eimp_final_input(eimp_B)
                ### convert complex numbers to float numbers
                Eimp_final = np.array(np.real(Eimp_final),dtype=float)
                ctqmc_mu = np.array(np.real(ctqmc_mu),dtype=float)

                params["hloc"]['one body'] = np.round(Eimp_final).tolist()

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
                tempmat = np.ones(Eimp_final.shape[0]*2)
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

        pass



    
    

    def CTQMCRun(self):
        
        #run_cmd = 'mpirun -np 1 '+diage_path+'/ComCTQMC/bin/CTQMC params'
        run_cmd = 'mpirun -np 4 ~/DiagE/ComCTQMC/bin/CTQMC params'
        print(run_cmd)
        
        with open('./ctqmc.out', 'w') as logfile, open('./ctqmc.err', 'w') as errfile:
            ret = subprocess.call(run_cmd, shell=True,stdout = logfile, stderr = errfile)
            if ret != 0:
                print("Error in CTQMC. Check ctqmc.err for error message.")
                sys.exit()
        
        return None
    
    def CTQMCMeasure(self):
        
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


    # def FmixLocDyn(self,iter : int, mix : float, Fb : np.ndarray, Fm : np.ndarray)->np.ndarray:

    #     norb = Fb.shape[0]
    #     ns = Fb.shape[2]
    #     nft = Fb.shape[3]

    #     F_new = np.zeros((norb,norb,ns,nft),dtype=complex,order='F')

    #     if iter == 1:
    #         mix = 1.0

    #     for ift in range(nft):
    #         for js in range(ns):
    #             for iorb in range(norb):
    #                 for jorb in range(norb):
    #                     F_new[iorb,jorb,js,ift] = mix*Fb[iorb,jorb,js,ift] + (1-mix)*Fm[iorb,jorb,js,ift]

    #     return F_new
    

    def CTQMCPostProcessing(self,iter,key,equiv,utilde_rf): # key -> problem number
    
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
        
        # Sigma_bare = self.FgaussianLocDyn(self.ft.omega,Sigma_bare,0.05,1/self.ft.beta,cutoff)
    
        # Green = self.FgaussianLocDyn(self.ft.omega,Green,0.05,1/self.ft.beta,cutoff)









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
        ndim = int(np.sqrt(len(obsjson["occupation-susceptibility-bulla"])))
        norbc = self.crystal.fprojector.shape[1]
        ns = self.crystal.ns
        nspin = 2             ### nspin could be different from ns in CTQMC

        nft = len(obsjson["occupation-susceptibility-bulla"]['0_0']['function'])

        tempmat = np.zeros((norbc,norbc,norbc,norbc,nspin,nspin,nft), dtype=np.complex128, order='F')

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb, ispin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb, jspin] = self.crystal.indexing(ndim,2,[norbc,nspin],0,ind2,nn2)
                name = str(ind1)+'_'+str(ind2)
                tempmat[iorb,jorb,jorb,iorb,ispin,jspin,:] = obsjson["occupation-susceptibility-bulla"][name]["function"]

        susceptibility = np.copy(tempmat)

        ### compute Pi_edmft using susceptibility and utilde
        Pi = PolImp(self.crystal,self.ft)
        Pi.add_key(susceptibility, utilde_rf, int(key)-1)






        print("******************************")
        print("Impurity Postprocessing Finish")
        print("******************************")


        return Sigma_h, Sigma_f, Sigma_c, Pi   ## return classes - (SigmaHImp, SigmaFImp, SigmaIGWC, PolIGW)








    def GreenInt_EDMFT(self, gbare : GreenBare, sigmah_gw : SigmaHartree, sigmaf_gw : SigmaFock, sigmac_gw : SigmaGWC
                               ,sigmah_edmft : np.ndarray, sigmaf_edmft : np.ndarray, sigmac_edmft : np.ndarray
                               ,sigmah_dc : np.ndarray, sigmaf_dc : np.ndarray, sigmac_dc : np.ndarray):

        sigmah_k = sigmah_gw.k + sigmah_edmft - sigmah_dc
        sigmaf_k = sigmaf_gw.k + sigmaf_edmft - sigmaf_dc
        sigmac_kf = sigmac_gw.kf + sigmac_edmft - sigmac_dc  ## should the sigmac_gw.rf needs to be converted from sigmac_gw.kf ?

        green_edmft = GreenInt_EDMFT(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah_k,sigmaf=sigmaf_k,sigmagwc=sigmac_kf)
        # green_edmft = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah_k,sigmaf=sigmaf_k,sigmagwc=sigmac_kf)

        # N = self.green.NumOfE(self.green.mu)
        # print("NumofE -- GW             :",N)
        N = green_edmft.NumOfE(green_edmft.mu)
        print("NumofE -- after embedding:",N)
        # print("Chemical potential -- GW             :",self.green.mu)
        print("Chemical potential -- after embedding:",green_edmft.mu)
        print("====")
        # print(self.green.kf[:,:,0,0,0])
        print(green_edmft.kf[:,:,0,0,0])
        # print(self.green.rf[:,:,0,0,0])
        print(green_edmft.rf[:,:,0,0,0])
        

        return green_edmft
    

    def W_EDMFT(self, vbare : VBare, pol_gw : PolLat, pol_edmft : np.ndarray, pol_dc : np.ndarray):

        pol_kf = pol_gw.kf + pol_edmft - pol_dc

        # w_edmft = pol_gw.Dyson(mat1, mat2)
        w_edmft = WLat(crystal=self.crystal,ft=self.ft,pol=pol_kf,vbare=vbare,c=self.c)

        return w_edmft
    
    def GW_update(self,g_edmft,vbare):

        sigmah = SigmaHartree(crystal=self.crystal,occ=g_edmft.occ,vbare=vbare.k)
        sigmaf = SigmaFock(crystal=self.crystal,occr=g_edmft.occr,vbare=vbare.r)
        pol = PolLat(crystal=self.crystal,ft=self.ft,green=g_edmft.rt)
        # pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
        w = WLat(crystal=self.crystal,ft=self.ft,pol=pol.kf,vbare=vbare,c=self.c)
        sigmagwc = SigmaGWC(crystal=self.crystal,ft=self.ft,green=g_edmft.rt,wlat=w.crt)
        # sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)

        return sigmah,sigmaf,sigmagwc,pol














    def SCFCheck(self, mat1 : np.ndarray, mat2 : np.ndarray):

        check = 0
        tempmat = abs(mat1-mat2)
        check = tempmat.max()
        return check

    def TightBinding(self, hopping : dict = None, onsite : dict = None, spin : bool = False, site : bool = False, valley : bool = False, fn : str = 'glob.h5'):

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        print('toto')
        print('dodo')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        if (hopping == None):
            print(errmessage)
            sys.exit()
        # niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist,hdf5file=tb)
        niham = NIHamiltonian(crystal=self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,hdf5file=fn,group=group)
        self.niham = niham
        # file.close()

        return None

    def HartreeFock(self, itermax : int, mix : float, hopping : dict = None,mode : str = "FromScratch", onsite : dict = None, spin : bool = False, valley : bool = False, avalley : bool = False, site : bool = False, asite : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None, ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, fn : str = 'glob.h5', group : str = 'hf'):

        errmessage = "missing input for HF calculation"
        if (hopping==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()
        
        if (mode == 'FromScratch'):
            
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=fn,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=fn,group=group)
            self.vbare = vbare
        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,hdf5file=fn,group=group)
            vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=fn,group=group)
            self.vbare = vbare





        for iter in range(1, itermax+1):
            if iter==1:
                # onsite_temp = None
                # if self.crystal.ns == 2:
                #     onsite_temp = {}
                #     for js in range(self.crystal.ns):
                #         onsite_temp[js] = {}
                #         for iorb in range(len(self.crystal.find)):
                #             ii, m1 = self.crystal.FAtomOrb(iorb)
                #             if js == 0:
                #                 onsite_temp[js][(ii,m1)] = 1.0
                #             else:
                #                 onsite_temp[js][(ii,m1)] = -1.0 
                if mode == "FromScratch":
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=fn,group='test_hf', avalley=avalley, asite=asite)
                    hold = Hamiltonian(crystal=self.crystal,ham=niham_temp.k,beta=self.ft.beta,hdf5file=fn,group=group)
                    hcheck = hold
                elif mode == "Restart":
                    niham_temp = NIHamiltonian(self.crystal,hopping=hopping,onsite=onsite,spin=spin,valley=valley,site=site,aferro=aferro, hdf5file=None,group='test_hf', avalley=avalley, asite=asite)
                    glob = h5py.File(fn,'r')
                    hf = glob['hf']
                    hk = hf['Hamiltonian']['hk'][:]
                    glob.close()
                    hold = Hamiltonian(crystal=self.crystal,ham=hk,beta=self.ft.beta,hdf5file=fn,group=group)
                    hcheck = hold
                    
                    

                hartreeold = None
                fockold = None

            print(hold.occ)
            sigmah = SigmaHartree(crystal=self.crystal,occ=hold.occ,vbare=vbare.k,hdf5file=fn,group=group)
            # sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            if (iter % 50 == 0):
                sigmah.Save(f'sigh.{iter}')
            sigmaf = SigmaFock(crystal=self.crystal,occr=hold.occr,vbare=vbare.r,hdf5file=fn,group=group)
            # sigmaf.k = sigmaf.Mixing(iter=iter,mix=mix,Fb=sigmaf.k,Fm=fockold)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigf.{iter}')
            hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=sigmah.k,sigmaf=sigmaf.k,hdf5file=fn,group=group)
            # hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=None,sigmaf=sigmaf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save(f'hk.{iter}')
            print(f"iteration : {iter}")
            if (iter % 10 == 0):
                fcheck = self.SCFCheck(hnew.kbare,hcheck.kbare)
                mucheck = abs(hnew.mu-hold.mu)
                hcheck = hnew
                print(f"criteria : {fcheck}\nchemical potential : {hnew.mu}")
            else:
                fcheck = 1
                mucheck = 1
            if (fcheck<=1.0e-9)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
                break
            elif(iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
            else:
                # hnew.OccMixing(iter=iter, mix=mix, occkb = hnew.occk, occkm=hold.occk)
                hnew.HMixing(iter=iter, mix=mix, hm=hold.k)
                hold=hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()


    def GWApproximation(self, itermax : int, mix : float, hoppinglist : list = None, onsitelist : list = None, spin : bool = False, valley : bool = False, site : bool = False, aferro : bool = False, loccoulomb : dict = None, nonloccoulomb : list = None,ohno : bool = False, jth : bool = False, ohnoyuka : bool = False, hdf5file : str = 'glob.h5', group : str = 'gw'):

        errmessage = "missing input for GW calculation"
        if (hoppinglist==None):
            print(errmessage)
            sys.exit()
        elif (loccoulomb==None):
            print(errmessage)
            sys.exit()

        niham = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,hdf5file=hdf5file,group=group)
        gbare = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham.k,hdf5file=hdf5file,group=group)
        vbare = VBare(crystal=self.crystal,orboption=loccoulomb,intamp=nonloccoulomb,ohno=ohno,jth=jth,ohnoyuka=ohnoyuka,hdf5file=hdf5file,group=group)
        self.vbare = vbare

        for iter in range(1,itermax+1):
            if iter == 1:
                # niham_temp = NIHamiltonian(crystal=self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin, valley=valley, hdf5file=hdf5file,group='test') 
                niham_temp = NIHamiltonian(self.crystal,hopping=hoppinglist,onsite=onsitelist,spin=spin,aferro=aferro, valley=valley,site=site,hdf5file=hdf5file,group='test_gw')
                gbare_temp = GreenBare(crystal=self.crystal,ft=self.ft,hamtb=niham_temp.k,hdf5file=hdf5file,group='test') 
                gold = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare_temp.kf,hdf5file=hdf5file,group=group)
                pkfold = None
                ckfold = None
                wold = 0
                # gbare.Save('gbare')


            print(gold.occ)
            print("Hartree calculation start")
            sigmah = SigmaHartree(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmah.Save(f'sigmah.{iter}')
            print("Hartree calculation finish")
            print("Fock calculation start")
            sigmaf = SigmaFock(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigmaf.{iter}')
            print("Fock calculation finish")
            print("Polarizability calculation start")
            pol = PolLat(crystal=self.crystal,ft=self.ft,green=gold.rt,hdf5file=hdf5file,group=group)
            pol.kf = pol.Mixing(iter=iter,mix=mix,Bb=pol.kf,Bold=pkfold)
            if (iter % 50 == 0):
                pol.Save(f'pkf.{iter}')
            print("Polarizability calculation finish")
            print("Screened coulomb interaction calculation start")
            w = WLat(crystal=self.crystal,ft=self.ft,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            print("Screened coulomb interaction calculation finish")
            print("GW self-energy calculation start")
            sigmagwc = SigmaGWC(crystal=self.crystal,ft=self.ft,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            sigmagwc.kf = sigmagwc.Mixing(iter=iter,mix=mix,Fb=sigmagwc.kf,Fm=ckfold)
            if (iter % 50 == 0):
                sigmagwc.Save(f'sigmagwckf.{iter}')
            print("GW self-energy calculation finish")
            print("GW green's function calculation start")
            gnew = GreenInt(crystal=self.crystal,ft=self.ft,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            if (iter % 50 == 0):
                gnew.Save(f'gkf.{iter}')
            print("GW green's function calculation start")

            fcheck = self.SCFCheck(gnew.kf,gold.kf)
            # bcheck = self.SCFCheck(w.kf,wold)
            mucheck = abs(gnew.mu-gold.mu)

            print(f"iteration : {iter} \nfcriteria : {fcheck} \nchemicalpotential : {gnew.mu}")

            if (fcheck <=1.0e-6)and(mucheck<=0.01):
                print(f"Self-consistency is achived with {iter}-th")
                self.greenbare = gbare
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
                break
            elif (iter==itermax):
                print(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.greenbare = gbare
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
            else:
                gold = gnew
                ckfold = sigmagwc.kf
                pkfold = pol.kf
                wold = w.kf

                del gnew, sigmah, sigmaf, sigmagwc, pol, w
                gc.collect()
            
            # self.green = gbare_temp
