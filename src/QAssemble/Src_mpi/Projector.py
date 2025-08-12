import numpy as np
from .MPIManager import MPIManager, FLatDynMPI
import h5py

class Projector(object):

    def __init__(self, mpimanager : MPIManager):


        # self.ns = crystal.ns
        self.mpimanager = mpimanager
        self.probspace = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.nspace = len(mpimanager.crystal.basisc)
        self.forb = 0
        self.borb = 0

        # if (projector != None):
        #     self.Cal(projector)

        # return None
    
    # def Cal(self, projector : dict):
        
    #     for key, val in projector.items():
    #         for orblist in val:
    #             atom = 0
    #             for orb in orblist:
    #                 if orb == orblist[0]:
    #                     atom = orb[0]
    #                 if atom!= orb[0]:
    #                     print("Different atoms are involved in the same space")
    #                     sys.exit()
    #         self.probspace[key] = [self.nspace + i for i in range(len(val))]
    #         self.nspace += len(val)
    #         self.fimpdict[key] = []
    #         for orblist in val:
    #             templist = []
    #             for orb in orblist:
    #                 find = self.cry.FIndex(orb)
    #                 templist.append(find)
    #             self.fimpdict[key].append(templist)
        
    #     for val in self.fimpdict.values():
    #         for orb in val:
    #             if (len(orb) > self.forb):
    #                 self.forb = len(orb)

    #     for key, val in self.fimpdict.items():
    #         self.bimpdict[key] = []
    #         for orb in val:
    #             templist = []
    #             for iorb in orb:
    #                 for jorb in orb:
    #                     a, _ = self.cry.FAtomOrb(iorb)
    #                     b, _ = self.cry.FAtomOrb(jorb)
    #                     if (a == b):
    #                         bind = self.cry.bbasis[iorb, jorb]
    #                         templist.append(bind)
    #             self.bimpdict[key].append(templist)

    #     for val in self.bimpdict.values():
    #         for orb in val:
    #             if (len(orb) > self.borb):
    #                 self.borb = len(orb)

    #     fprojector = np.zeros((len(self.cry.find), self.forb, self.ns, self.nspace), dtype=np.float64, order='F')
    #     bprojector = np.zeros((len(self.cry.bind), self.borb, self.ns, self.nspace), dtype=np.float64, order='F')


    #     for js in range(self.ns):
    #         for key, val in self.probspace.items():
    #             for ii, ispace in enumerate(val):
    #                 for ind in self.fimpdict[key][ii]:
    #                     fprojector[ind,self.fimpdict[key][ii].index(ind),js,ispace] = 1.0

    #     for js in range(self.ns):
    #         for key, val in self.probspace.items():
    #             for ii, ispace in enumerate(val):
    #                 for ind in self.bimpdict[key][ii]:
    #                     bprojector[ind,self.bimpdict[key][ii].index(ind),js,ispace] = 1.0

    #     self.fprojector = fprojector
    #     self.bprojector = bprojector

    #     return None
    
    # # def ReadEquivMat(self, imp : dict):

    # #     nprob = len(self.probspace)
        
    # #     if (len(imp) -1 != nprob):
    # #         print("***** number of impurity problems are not the same *****")
    # #         print("***** program stopped!!! *****")
    # #         sys.exit()

    # #     self.impindex = []

    # #     for ii in range(nprob):
    # #         iimp = str(ii+1)
    # #         N = len(imp[iimp]['impurity_matrix'])

    # #         equivmat = imp[iimp]['impurity_matrix']

    # #         for i in range(N):
    # #             for j in range(N):
    # #                 if (equivmat[i, j] != 0):
    # #                     if (equivmat[i, j] > len(self.impindex[ii])):
    # #                         for k in range((equivmat[i, j] - len(self.impindex[ii]))):
    # #                             self.impindex[ii].append([])
    # #                         self.impindex[ii][equivmat[i,j] - 1].append([i, j])
    # #                     else:
    # #                         self.impindex[ii][equivmat[i,j] - 1].append([i, j])

    # def Load(self, imp : list):

    #     from collections import defaultdict
    #     tempdict = defaultdict(list)

    #     for ridx, row in enumerate(imp):
    #         for cidx, value in enumerate(row):
    #             if (value != 0):
    #                 tempdict[value].append((ridx, cidx))

    #     impdict = {}
    #     i = 1
    #     for value in tempdict.values():
    #         impdict[str(i)] = []
    #         templist = []
    #         for val in value:
    #             (a, m) = self.cry.FAtomOrb(val[0])
    #             templist.append([a, m])
    #         impdict[str(i)].append(templist)

    #         i += 1
        
    #     # self.Cal(impdict)

    #     return impdict
        
    def Load(self, path : str):

        glob = h5py.File(path+'/global.dat')
        nbndf = glob['full_space']['gw']['nbndf'][:]
        includebands = glob['combasis_fermion']['include_bands'][:]
        numwann = glob['combasis_fermion']['num_wann']
        correlated = glob['combasis_boson']['wan_correlated'][:]

        nk = self.mpimanager.crystal.nk
        nf = glob['full_space']['gw']['n_omega'][:]
        ntau = glob['full_space']['gw']['n_tau'][:]
        nprock = glob['comweiss_fermion']['nproc_k'][:]
        nprocf = glob['comweiss_fermion']['nproc_w'][:]
        glob.close()

        nodedict = self.mpimanager.mpidict[(nk, nf, ntau, nprock, nprocf)]
        commk = nodedict['commk']
        rank = commk.Get_rank()
        nk_loc = len(self.mpimanager.klocal2global[rank])

        fpfl = np.zeros((nbndf[0], numwann[0], nk_loc, self.mpimanager.crystal.ns), dtype=np.complex128, order='F')
        fplc = np.zeros((numwann[0], len(correlated), self.mpimanager.crystal.ns, self.nspace), dtype=np.complex128, order='F')

        for js in range(self.mpimanager.crystal.ns):
            for j in range(numwann[0]):
                for i in includebands:
                    for ik in range(nk_loc):
                        fpfl[i, j, ik, js] = 1.0

        for ispace in range(self.nspace):
            for js in range(self.mpimanager.crystal.ns):
                for j in correlated:
                    for i in range(numwann[0]):
                        fplc[i, j, js, ispace] = 1.0

        self.fpfl = fpfl
        self.fplc = fplc

        return None
                