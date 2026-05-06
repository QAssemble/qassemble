import string as string
from typing import Any
import matplotlib as mat
import re as re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.font_manager as fm
import json, os, shutil, sys
import itertools
import logging
import scipy.optimize
from sympy.physics.wigner import gaunt, wigner_3j
import scipy.linalg
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
import subprocess
import copy
# import Crystal, FTGrid
from .Crystal import Crystal
from .Projector import Projector
from .utility.Common import Common
from .utility.Dyson import Dyson
from .utility.Projection import Projection as PJ

logger = logging.getLogger("QAssemble")


class BLocStc(object):

    def __init__(self,crystal : Crystal, projector : Projector = None):

        self.crystal = crystal
        self.projector = projector

    def _require_projector(self) -> Projector:
        if self.projector is None:
            raise ValueError("projector is required for impurity/local index mapping")
        return self.projector

    def ResolveProblemKey(self, key):
        if self.projector is None:
            raise ValueError("projector is required to resolve problem key")

        pkey = key if key in self.projector.bprojector else str(key)
        if pkey not in self.projector.bprojector:
            raise KeyError(f"Unknown bosonic impurity problem key '{key}'")
        return pkey

    def Projection(self, matin : np.ndarray, key) -> np.ndarray:
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 4:
            raise ValueError(f"matin must be 4D, got {matin.ndim}D")

        pkey = self.ResolveProblemKey(key)
        return PJ.BLocStc(matin, self.projector.bprojector[pkey])

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex128)

        
        tempmat = self.crystal.OrbSpin2Composite(matin)
        tempmat2 = np.linalg.inv(tempmat)
        matout = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout
    
    def Mixing(self, iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray)-> np.ndarray:

        norb = Bb.shape[0]
        ns = Bb.shape[2]

        Bnew = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')
        
        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew

    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]

        projector = self._require_projector()
        probindex = projector.probindex if projector.probindex else projector.probspace

        nspace = 0
        for val in probindex.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')

        for key, val in probindex.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
        
    def Loc2Imp(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]

        projector = self._require_projector()
        probindex = projector.probindex if projector.probindex else projector.probspace

        nspace = 0
        for val in probindex.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')

        for key, val in probindex.items():
            iprob = int(key)-1
            for ispace in val:
                matloc[...,ispace] = matimp[...,iprob]

        return matloc
    
    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:

        ns = matin.shape[2]
        nind = np.amax(equiv)
        matdict = {}

        
        for ind in range(nind):
            matdict[ind+1] = []
            pos = Common.FindPositions(equiv,ind+1)
            for js in range(ns):
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii,jj,js,ks]
                    e /= len(pos)
                    matdict[ind+1].append(e)
        
        return matdict
    
    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        matout = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = Common.FindPositions(equiv,ind+1)
                    for ii,jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def ReadDict(self, equiv : np.ndarray, mat_dict : dict) -> np.ndarray:
        """Read equivalent-orbital dict data as a local static bosonic matrix."""
        norb = len(equiv)
        ns = self.crystal.ns
        mat_out = np.zeros((norb,norb,ns,ns), dtype=np.complex128, order='F')
        nind = int(np.amax(equiv))

        for js in range(ns):
            for ks in range(ns):
                for ind in range(1, nind + 1):
                    key = str(ind) if str(ind) in mat_dict else ind
                    if key not in mat_dict:
                        continue
                    val = mat_dict[key]
                    pos_row, pos_col = np.where(equiv == ind)
                    for ii, jj in zip(pos_row, pos_col):
                        mat_out[ii,jj,js,ks] = val

        return mat_out

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = mat1.shape[2]

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')

        matout = Dyson.BLocStc(mat1,mat2)

        return matout

    # def Embedding(self, matin : np.ndarray):

    #     norb = len(self.crystal.bind)
    #     ns = self.crystal.ns
    #     nrk = len(self.crystal.kpoint)
    #     nspace = self.crystal.bprojector.shape[3]

    #     matout = np.zeros((norb,norb,ns,ns,nrk),dtype=np.complex128,order='F')

    #     for ispace in range(nspace):
    #         matout += QAFort.embedding.blocstc(nrk,matin[...,ispace],self.crystal.bprojector.shape[...,ispace])

    #     return matout
    
    def Double2Quad(self, matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex128,order='F')
        
        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Double2Quad(matin[...,js,ks])

        return matout
    
    def Quad2Double(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Quad2Double(matin[...,js,ks])

        return matout
    
    def Double2Full(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc**2,norbc**2,ns,ns),dtype=np.complex128)

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.Double2Full(matin[...,js,ks])
        
        return matin
    
    def Full2Double(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Full2Double(matin[...,js,ks])

        return matout
    
    def Quad2Full(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc*norbc,norbc*norbc,ns,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.Quad2Full(matin[...,js,ks])
        
        return matout
    
    def Full2Quad(self,matin):

        norb = len(self.crystal.bind)
        norbc = len(self.crystal.find)
        ns = self.crystal.ns

        matout = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            for ks in range(ns):
                matout[...,js,ks] = self.crystal.Full2Quad(matout[...,js,ks])

        return matout
    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]

        if os.path.exists('blocstc'):
            pass
        else:
            os.mkdir('blocstc')
        os.chdir('blocstc')
        
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, js, Re(B), Im(B)\n")
            for ks in range(ns):
                for js in range(ns):
                    for jorb in range(norb):
                        for iorb in range(norb):
                            f.write(f"{iorb} {jorb} {js} {ks} {matin[iorb,jorb,js,ks].real} {matin[iorb,jorb,js,ks].imag}\n")
        
        os.chdir("..")
        return None

class VLoc(BLocStc):

    def __init__(self, crystal: Crystal, voption : dict = None, projector : Projector = None):
        super().__init__(crystal, projector)
        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        self.onsitelist = None
        self.vloc = np.zeros((norb,norb,ns,ns),dtype=float,order='F')
        if voption is None:
            logger.error("voption does not exist")
            sys.exit()

        self.parameter = {}
        self.SetLocalInteracting(voption)
        # self.GenOnsite()

    def _format_slater_values(self, l: int, value: list) -> dict:
        labels = ("F0", "F2", "F4", "F6")
        # Slater expansion uses up to l+1 radial integrals (k = 0, 2, 4, 6).
        max_n = max(1, min(4, int(l) + 1))
        vals = {"F0": 0.0, "F2": 0.0, "F4": 0.0, "F6": 0.0}
        for ii in range(min(len(value), max_n)):
            vals[labels[ii]] = float(value[ii])
        return vals

    def _store_slater_parameter(self, atom: int, l: int, value: list, orbitals: list) -> None:
        if "slater_by_atom" not in self.parameter:
            self.parameter["slater_by_atom"] = {}
        self.parameter["slater_by_atom"][str(int(atom))] = {
            "l": int(l),
            "orbitals": [int(orb) for orb in orbitals],
            "values": self._format_slater_values(l=l, value=value),
        }

    def GetSlaterByProblem(
        self,
        key,
        projector: Projector = None,
        include_defaults: bool = True,
    ) -> dict:
        default = {"F0": 0.0, "F2": 0.0, "F4": 0.0, "F6": 0.0}
        slater_by_atom = self.parameter.get("slater_by_atom", {})
        if len(slater_by_atom) == 0:
            return copy.deepcopy(default) if include_defaults else {}

        projector = projector if projector is not None else self.projector
        if projector is None:
            return copy.deepcopy(default) if include_defaults else {}

        key = str(key)
        if key not in projector.probspace:
            raise KeyError(f"Unknown impurity problem key '{key}'")

        candidates = []
        for atom in projector.probspace[key]:
            atom_key = str(int(atom))
            if atom_key in slater_by_atom:
                candidates.append(slater_by_atom[atom_key]["values"])

        if len(candidates) == 0:
            return copy.deepcopy(default) if include_defaults else {}

        ref = candidates[0]
        for cand in candidates[1:]:
            if any(abs(float(cand[name]) - float(ref[name])) > 1.0e-12 for name in ("F0", "F2", "F4", "F6")):
                raise ValueError(
                    f"Problem '{key}' has multiple atoms with different Slater parameters; "
                    "define per-problem Slater mapping explicitly."
                )

        out = copy.deepcopy(default) if include_defaults else {}
        out.update(ref)
        return out

    def SetLocalInteracting(self,voption : dict):
        
        ns = self.crystal.ns

        if voption["Parameter"] == "Kanamori":
            
            
            for key, val in voption["option"].items():
                atom = int(key-1)
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    logger.error("Invalid l value set")
                    sys.exit()
                
                tempmat = self.KanamoriParameter(norb=norbc,val=val["value"])
                for js in range(ns):
                    for ks in range(ns):
                        for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                            iorb = self.crystal.BIndex([atom,[m1,m4]])
                            jorb = self.crystal.BIndex([atom,[m2,m3]])
                            if (iorb is not None)and(jorb is not None):
                                self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        if voption["Parameter"] == "Slater":
            for key, val in voption["option"].items():
                atom = int(key)-1
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    logger.error("Invalid l value set")
                    sys.exit()
                self._store_slater_parameter(
                    atom=atom,
                    l=val["l"],
                    value=val["value"],
                    orbitals=val["orbitals"],
                )
                tempmat = self.SlaterParameter(l=val["l"],norbc=norbc,val=val["value"])
                for js, ks in itertools.product(list(range(ns)),list(range(ns))):
                    for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                        iorb = self.crystal.BIndex([atom,[m1,m4]])
                        jorb = self.crystal.BIndex([atom,[m2,m3]])
                        if (iorb is not None)and(jorb is not None):
                            self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]
        if voption["Parameter"] == "SlaterKanamori":
            for key, val in voption["option"].items():
                atom = int(key-1)
                norbc = len(val["orbitals"])
                if norbc > len(self.crystal.find):
                    logger.error("Invalid l value set")
                    sys.exit()
                tempmat = self.SlaterKanamori(l=val["l"],norb=norbc,val=val["value"])
                for js, ks in itertools.product(list(range(ns)),list(range(ns))):
                    for m1,m2,m3,m4 in itertools.product(val["orbitals"],val["orbitals"],val["orbitals"],val["orbitals"]):
                        iorb = self.crystal.BIndex([atom,[m1,m4]])
                        jorb = self.crystal.BIndex([atom,[m2,m3]])
                        if (iorb is not None)and(jorb is not None):
                            self.vloc[iorb,jorb,js,ks] = tempmat[m1,m2,m3,m4,js,ks]

        
        return None
    
    def GenOnsite(self):
        
        norbc = len(self.crystal.find)
        ns = self.crystal.ns
        onsitelist = []
        
        tempmat = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=np.complex128,order='F')

        tempmat = self.Double2Quad(self.vloc)

        for js in range(ns):
            for ks in range(ns):
                for iorbc in range(norbc):
                    if (js==ks):
                        onsitelist.append(-tempmat[iorbc,iorbc,iorbc,iorbc,js,ks])
        
        self.onsitelist = onsitelist

        return None

    
    def KanamoriParameter(self, norb : int, val : list) -> np.ndarray:

        # print("Warning : In kanamori interaction, self interaction term has been added")
        ns = self.crystal.ns
        v = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')
        U = val[0]
        Up = val[1]
        J = val[2]

        for js in range(ns):
            for ks in range(ns):
                for m1 in range(norb):
                    for m2 in range(norb):
                        for m3 in range(norb):
                            for m4 in range(norb):
                                if (m1==m2==m3==m4)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = U
                                elif(m1==m4)and(m2==m3)and(m1!=m2)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = Up
                                elif(m1==m4)and(m2==m3)and(m1!=m2)and(js==ks):
                                    v[m1,m2,m3,m4,js,ks] = Up-J
                                elif (m1==m3)and(m2==m4)and(m1!=m2)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = J
                                elif (m1==m2)and(m3==m4)and(m1!=m3)and(js!=ks):
                                    v[m1,m2,m3,m4,js,ks] = J
        v *= 0.5
        return v


    def SlaterParameter(self, l : int = None,norbc : int=None, val : list=None, sc : str = 'c') -> np.ndarray:
        
        # error message
        # print("Only calculate the odd number of orbitals")    
        ns = self.crystal.ns
        norb = 2*l+1
        vtemp = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')
        v = np.zeros((norbc,norbc,norbc,norbc,ns,ns),dtype=float,order='F')

        # l = int((norb-1)/2)
        m = list(range(-l,l+1))

        for n, f in enumerate(val):
            k = 2*n
            
            for js in range(ns):
                for ks in range(ns):
                    for m1 in m:
                        for m2 in m:
                            for m3 in m:
                                for m4 in m:
                                    vtemp[m1+l,m2+l,m3+l,m4+l,js,ks] += f*self.AngularIntegral(l,k,m1,m2,m4,m3)
        if sc == 'c':
            for js in range(ns):
                for ks in range(ns):
                    tempmat = vtemp[:,:,:,:,js,ks]
                    tempmat2 = self.Spherical2Cubic(tempmat,l)
                    vtemp[:,:,:,:,js,ks] = tempmat2
            if (l==2)and(norbc==3):
                for ii, iorbc in enumerate([0,1,3]):
                    for jj, jorbc in enumerate([0,1,3]):
                        for kk, korbc in enumerate([0,1,3]):
                            for ll, lorbc in enumerate([0,1,3]):
                                v[ii,jj,kk,ll] = vtemp[iorbc,jorbc,korbc,lorbc]
            elif (l==2)and(norbc==2):
                for ii, iorbc in enumerate([2,4]):
                    for jj, jorbc in enumerate([2,4]):
                        for kk, korbc in enumerate([2,4]):
                            for ll, lorbc in enumerate([2,4]):
                                v[ii,jj,kk,ll] = vtemp[iorbc,jorbc,korbc,lorbc]
            else:
                v = vtemp
            return v
        else:
            return v
    
    
    def SlaterKanamori(self,l : int,norb : int, val : list) -> np.ndarray :

        U = val[0]
        Up = val[1]
        J = val[2]
        ratio = 0.625
        ns = self.crystal.ns
        # print(norb)

        v = np.zeros((norb,norb,norb,norb,ns,ns),dtype=float,order='F')

        if norb == 1:
            F0 = U
            F2 = 0
            F4 = 0
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
            return v
        if norb == 3:
            F2 = 441/(27+20*ratio)*J
            F4 = ratio*F2
            F0 = U-4/49*(F2+F4)
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
                
            return v
        if norb == 5:
            # F2 = 14/(1+ratio)*J
            # F4 = ratio*J
            # F0 = U
            F0 = U-8/5*J
            F2 = 49*(1/4+1/7)*J
            F4 = 63/5*J
            v = self.SlaterParameter(l,norb,[F0,F2,F4])
            return v
        
    def AngularIntegral(self,l,k,m1,m2,m3,m4):

        ang_int = 0
        pi = np.pi

        for q in range(-k,k+1):
            ang_int += gaunt(l,k,l,-m1,q,m3)*np.conjugate(gaunt(l,k,l,m4,-q,-m2))*((-1.0 if(m1+q+m2)%2 == 1 else 1.0))

        ang_int *= 4*pi/(2*k+1)

        return ang_int

    def RotationMatrix(self,l : int):

        mrange = int(2*l+1)
        R = np.zeros((mrange,mrange),dtype=np.complex128)
        
        if l == 0:
            R = np.eye(mrange,mrange,dtype=np.complex128)
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
    
    def Spherical2Cubic(self,v : np.ndarray,l : int):
        
        
        R = self.RotationMatrix(l)
        Rdag = np.conjugate(np.transpose(R))
        
        tempmat = np.einsum("ab,cd,bdeg,ef,gh",Rdag,Rdag,v,R,R)
        tempmat = np.real(tempmat)
        
        V = np.array(tempmat,dtype=float,order='F')
        
        return V


    def GetUijklComCTQMC(self, key):

        norb = len(self.crystal.find)
        ns = self.crystal.ns

        projector = self._require_projector()
        orb = projector.fimpdict[key][0]
        norbc = len(orb)
        tempmat = np.zeros((norb, norb, norb, norb), dtype=np.complex128, order='F')
        vloc = np.zeros((norbc, norbc, norbc, norbc, ns, ns), dtype=np.complex128, order='F')
        for ks in range(ns):
            for js in range(ns):
                tempmat = self.crystal.Double2Quad(self.vloc[...,js,ks])
                for ll, lorb in enumerate(orb):
                    for kk, korb in enumerate(orb):
                        for jj, jorb in enumerate(orb):
                            for ii, iorb in enumerate(orb):
                                vloc[ii, jj, kk, ll, js, ks] = tempmat[iorb, jorb, korb, lorb]

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
                                                    val = vloc[i, j, k, l, 0, 0].real
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
                                                    val = vloc[i, j, k, l, si, sj].real
                                                    val = abs(val)
                                                    if (val > 0.001):
                                                        U[idx] = val
                                                idx += 1
        else:
            logger.error("SOC is not False")
            sys.exit()
        self.u_ctqmc = U

        return U
