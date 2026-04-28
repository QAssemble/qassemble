import numpy as np
import logging
import os, sys
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .FLatStc import FLatStc
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.Projection import Projection as PJ

logger = logging.getLogger("QAssemble")

class FLocStc(object):

    def __init__(self,crystal : Crystal, projector : Projector):

        self.crystal = crystal
        self.projector = projector

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            matinv[:,:,js] = Common.MatInv(mat[:, :, js])

        return matinv
    
    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:

        norb = Fb.shape[0]
        ns = Fb.shape[2]

        Fnew = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Fold = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        Fnew = mix*Fb + (1.0-mix)*Fold

        return Fnew

    def _resolve_equiv_matrix(self, imp=None, key=None) -> np.ndarray:
        """Resolve an equivalent-orbital matrix from legacy/new impurity inputs.

        Supported inputs:
        - 2D ndarray/list: used directly as equivalence matrix.
        - 1D ndarray/list: interpreted as diagonal class labels and promoted via ``np.diag``.
        - Legacy dict: ``imp[str(key)]['impurity_matrix']``.
        - Direct dict: ``imp[str(key)]`` is the equivalence matrix itself.
        - Fallback: ``self.projector.equiv[str(key)]`` when ``imp`` is None.
        """
        def _resolve_dict_key(dct, key_):
            if key_ is None:
                if len(dct) == 1:
                    return str(next(iter(dct.keys())))
                raise ValueError(
                    "key is required when multiple impurity problems are present"
                )
            k_ = str(key_)
            if k_ not in dct:
                raise KeyError(f"equiv source does not contain key '{k_}'")
            return k_

        if imp is None:
            if self.projector is None or not isinstance(self.projector.equiv, dict):
                raise ValueError(
                    "imp is None and projector.equiv is not available; "
                    "provide imp or set projector.equiv"
                )
            peq = self.projector.equiv
            k = _resolve_dict_key(peq, key)
            equiv = np.asarray(peq[k])

        elif isinstance(imp, np.ndarray):
            equiv = imp
        elif isinstance(imp, (list, tuple)):
            equiv = np.asarray(imp)
        elif isinstance(imp, dict):
            k = _resolve_dict_key(imp, key)
            if isinstance(imp[k], dict):
                if "impurity_matrix" not in imp[k]:
                    raise KeyError(
                        f"imp['{k}'] must contain an 'impurity_matrix' entry"
                    )
                equiv = np.asarray(imp[k]["impurity_matrix"])
            else:
                equiv = np.asarray(imp[k])
        else:
            raise TypeError(
                "imp must be ndarray/list/tuple (equiv matrix), direct equiv dict, or legacy impurity dict"
            )

        if equiv.ndim == 1:
            equiv = np.diag(equiv)

        if equiv.ndim != 2 or equiv.shape[0] != equiv.shape[1]:
            raise ValueError(
                f"equivalence matrix must be square 2D, got shape {equiv.shape}"
            )

        return np.asarray(equiv, dtype=int)

    def Arr2Dict(self, equiv : np.ndarray, matin : np.ndarray) -> dict:
        """Average a local static fermionic matrix over equivalent orbital pairs."""
        if matin.ndim == 2:
            matin = matin[..., np.newaxis]
        elif matin.ndim != 3:
            raise ValueError(f"matin must be 2D or 3D, got {matin.ndim}D")

        norb = matin.shape[0]
        if matin.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin.shape}"
            )

        ns = matin.shape[2]
        if ns != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={ns}, crystal ns={self.crystal.ns}"
            )
        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        matdict = {}
        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue

            if ns == 1:
                e = 0.0 + 0.0j
                for ii, jj in pos:
                    e += matin[ii, jj, 0]
                matdict[str(ind)] = e / len(pos)
            else:
                e = []
                for js in range(ns):
                    temp = 0.0 + 0.0j
                    for ii, jj in pos:
                        temp += matin[ii, jj, js]
                    e.append(temp / len(pos))
                matdict[str(ind)] = e

        return matdict

    def Dict2Arr(self, equiv : np.ndarray, matdict : dict) -> np.ndarray:
        """Expand equivalent-orbital dict data back to local static fermionic matrix."""
        norb = len(equiv)
        nind = int(np.amax(equiv))
        ns = self.crystal.ns

        matout = np.zeros((norb, norb, ns), dtype=np.complex128, order='F')

        for ind in range(1, nind + 1):
            key = str(ind) if str(ind) in matdict else ind
            if key not in matdict:
                continue
            val = matdict[key]
            pos = Common.FindPositions(equiv, ind)

            if ns == 1:
                for ii, jj in pos:
                    matout[ii, jj, 0] = val
            else:
                val = np.asarray(val, dtype=np.complex128)
                if val.ndim != 1 or val.shape[0] != ns:
                    raise ValueError(
                        f"matdict['{ind}'] must be a 1D spin array of length {ns}"
                    )
                for js in range(ns):
                    for ii, jj in pos:
                        matout[ii, jj, js] = val[js]

        return matout

    def ReadDict(self, equiv : np.ndarray, mat_dict : dict) -> np.ndarray:
        """Read equivalent-orbital dict data as a local static fermionic matrix."""
        return self.Dict2Arr(equiv=equiv, matdict=mat_dict)

    def AverageByEquiv(self, equiv : np.ndarray, matin : np.ndarray, squeeze : bool = True) -> np.ndarray:
        """Average equivalent orbital classes and return array in one pass.

        Input:
        - matin: (norb, norb) or (norb, norb, ns)
        Output:
        - matout: same semantic shape as input (squeezed to 2D when ns=1 and squeeze=True)
        """
        if matin.ndim == 2:
            matin3 = matin[..., np.newaxis]
        elif matin.ndim == 3:
            matin3 = matin
        else:
            raise ValueError(f"matin must be 2D or 3D, got {matin.ndim}D")

        norb = matin3.shape[0]
        if matin3.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb or equiv.shape[1] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin3.shape}"
            )
        if matin3.shape[2] != self.crystal.ns:
            raise ValueError(
                f"spin dimension mismatch: matin ns={matin3.shape[2]}, crystal ns={self.crystal.ns}"
            )

        matout = np.array(matin3, dtype=np.complex128, copy=True, order='F')
        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue
            for js in range(self.crystal.ns):
                avg = 0.0 + 0.0j
                for ii, jj in pos:
                    avg += matin3[ii, jj, js]
                avg /= len(pos)
                for ii, jj in pos:
                    matout[ii, jj, js] = avg

        if squeeze and self.crystal.ns == 1:
            return matout[:, :, 0]
        return matout

    def AverageImpurityByEquiv(self, imp=None, matimp : dict = None, squeeze : bool = True) -> dict:
        """Average equivalent orbital classes for all impurity problems at once.

        Parameters
        ----------
        imp : dict | ndarray | list | tuple | None
            Equivalence source. If None, ``self.projector.equiv`` is used.
            Legacy impurity input ``imp[key]['impurity_matrix']`` is also supported.
        matimp : dict
            Problem-wise matrices: ``{key: ndarray}``, each ndarray is (norb,norb) or (norb,norb,ns).
        """
        if not isinstance(matimp, dict):
            raise TypeError("matimp must be dict keyed by impurity problem key")

        matout = {}
        for key, matin in matimp.items():
            equiv = self._resolve_equiv_matrix(imp=imp, key=key)
            matout[str(key)] = self.AverageByEquiv(equiv=equiv, matin=matin, squeeze=squeeze)

        return matout

    def imp_B2F(self, imp=None, B : np.ndarray = None, key = None) -> dict:
        """Legacy wrapper: average by equivalent-orbital classes."""
        if B is None:
            raise ValueError("B is required")
        equiv = self._resolve_equiv_matrix(imp=imp, key=key)
        return self.Arr2Dict(equiv=equiv, matin=B)

    def imp_F2B(self, imp=None, F : dict = None, key = None, squeeze : bool = True) -> np.ndarray:
        """Legacy wrapper: map equivalent-orbital dict back to matrix."""
        if F is None:
            raise ValueError("F is required")
        equiv = self._resolve_equiv_matrix(imp=imp, key=key)
        mat = self.Dict2Arr(equiv=equiv, matdict=F)
        if squeeze and mat.shape[2] == 1:
            return mat[:, :, 0]
        return mat
    
    
    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        return Dyson.FLocStc(mat1, mat2)

    
    def Save(self,matin : np.ndarray, fn : str):

        norb = matin.shape[0]
        ns = matin.shape[2]

        if os.path.exists('flocstc'):
            pass
        else:
            os.mkdir("flocstc")
        os.chdir("flocstc")
        with open(fn+'.txt','w') as f:
            f.write("iorb, jorb, is, Re(F), Im(F)\n")
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        f.write(f"{iorb} {jorb} {js} {matin[iorb,jorb,js].real} {matin[iorb,jorb,js].imag}\n")
        os.chdir("..")
        return None
    
    def Projection(self, matin : np.ndarray):
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 4:
            raise ValueError(f"matin must be 4D, got {matin.ndim}D")

        norb = matin.shape[0]
        ns = matin.shape[2]
        nrk = matin.shape[4]

        matdict = {}
        for key, proj in self.projector.fprojector.items():
            norbc = proj.shape[1]
            tempmat = np.zeros((norbc, norbc, ns, nrk), dtype=np.complex128, order='F')

            tempmat = PJ.FLatStc(matin, proj)

            
            matdict[key] = tempmat

        return matdict
    
    
class EImp(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, hamtb : np.ndarray, mu : float, sigh : np.ndarray = None, sigf : np.ndarray = None, hloc : dict = None, floc : dict = None):

        super().__init__(crystal, projector)

        self.hamtb = hamtb
        self.mu = mu
        self.ham = None
        self.sig = None
        self.e = {}

        tempmat = np.zeros_like(hamtb, dtype=np.complex128, order='F')

        for ik in range(hamtb.shape[3]):
            for js in range(hamtb.shape[2]):
                tempmat[...,js,ik] = hamtb[...,js,ik] - mu*np.eye(hamtb.shape[0], dtype=np.complex128)
        
        if sigh is not None:
            tempmat += sigh
        
        if sigf is not None:
            tempmat += sigf

        self.ham = tempmat

        if (hloc is not None) and (floc is not None):
            print("Double counting term entered.")
            tempmat2 = {}
            for key in hloc.keys():
                tempmat2[key] = hloc[key] + floc[key]

            self.sig = tempmat2 

        
        self.Cal()

    def Cal(self):

        e = self.Projection(self.ham)

        if (self.sig is not None):
            tempdict = e.copy()
            e = {}
            for key, mat in tempdict.items():

                mat -= self.sig[key]

                e[key] = mat

        
        self.e = e

        return None

    def Eimp_final_input(self, key, Eimp : np.ndarray = None):
        key = self.projector._require_problem(key)
        ns = self.crystal.ns
        norbc = self.projector.fprojector[key].shape[1]

        if Eimp is None:
            Eimp = self.e[key]

        Eimp = np.asarray(Eimp, dtype=np.complex128)
        if Eimp.ndim == 2:
            Eimp = Eimp[:, :, np.newaxis]
        if Eimp.ndim != 3:
            raise ValueError(f"Eimp must be 2D or 3D, got {Eimp.ndim}D")

        if ns==1:
            I = np.identity(norbc)
            A_final = np.zeros((norbc*2,norbc*2), dtype=np.complex128, order='F')

            ctqmc_mu = -Eimp[0,0,0]
            A = Eimp[...,0] + ctqmc_mu*I
            A_final[...] = np.kron(np.eye(2),A)
        
        elif ns==2:
            print("Nspin is not 1")
            sys.exit()
        
        return A_final,ctqmc_mu
            
            
            
class SigHLoc(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, occ : dict = None, vloc : dict = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.occ = occ
        self.vloc = vloc
        self.hloc = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

    
    def Cal(self):

        projector = self.projector.fprojector
        h = {}

        for key, proj in projector.items():
            norbc = proj.shape[1]
            ns = proj.shape[2]
            v = self.vloc[key]
            norb = v.shape[0]

            h[key] = np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

            if ns != 1:

                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind1, nn1)

                    iorbc1, iorbc2 = self.projector.ProbBorb2FPair(key, iorb)

                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind2, nn2)

                        iorbc3, iorbc4 = self.projector.ProbBorb2FPair(key, jorb)

                        h[key][iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[key][iorbc4, iorbc3, ks])
            else:
                if (self.crystal.soc == True):
                    C = 1
                else:
                    C = 2
                
                for ind1 in range(norb * ns):
                    nn1 = [0] * 2
                    ind1, [iorb, js] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind1, nn1)

                    iorbc1, iorbc2 = self.projector.ProbBorb2FPair(key, iorb)

                    for ind2 in range(norb * ns):
                        nn2 = [0] * 2
                        ind2, [jorb, ks] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind2, nn2)

                        iorbc3, iorbc4 = self.projector.ProbBorb2FPair(key, jorb)

                        h[key][iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[key][iorbc4, iorbc3, ks]) * C

        self.hloc = h





# class SigmaFLoc(FLocStc):

#     def __init__(self, crystal: Crystal, gloc : GreenLoc, vbare : object):
#         super().__init__(crystal)

#         self.gloc = gloc
#         self.vbare = vbare
#         self.floc = None
#         self.fimp = None
#         self.fdyn = None
    
#         self.Cal()
#         self.MakeDyn()

#     def Cal(self):
        
#         norbc = self.crystal.fprojector.shape[1]
#         ns = self.crystal.ns
#         norb = self.crystal.bprojector.shape[1]
#         nspace = self.crystal.fprojector.shape[3]

#         U = np.zeros((norb,norb,ns,ns,nspace),dtype=np.complex128,order='F')
#         floc = np.zeros((norbc,norbc,ns,nspace),dtype=np.complex128,order='F')
        

#         for ispace in range(nspace):
#             U[...,ispace] = QAFort.projection.blatstc(self.vbare.k,self.crystal.bprojector[...,ispace])

#             for js in range(ns):
#                 for iorb in range(norb):
#                     iorbc1, iorbc4 = self.crystal.b2f[iorb]
#                     for jorb in range(norb):
#                         iorbc3, iorbc2 = self.crystal.b2f[jorb]
#                         floc[iorbc1,iorbc2,js,ispace] += self.gloc.gf[iorbc4,iorbc3,js,-1,ispace]*U[iorb,jorb,js,js,ispace]

#         self.floc = floc
#         self.fimp = self.Loc2Imp(floc)
        
#         return None



# class SigmaFImp(FLocStc):

#     def __init__(self, crystal: Crystal):
#         super().__init__(crystal)
#         self.Cal()

#     def Cal(self):
#         pass
