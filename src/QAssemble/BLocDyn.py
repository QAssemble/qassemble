import numpy as np
import logging
import sys
import json
import scipy.optimize
import scipy.linalg.lapack
import copy
import h5py
import time, datetime
from .Crystal import Crystal
from .BLocStc import VLoc
from .Projector import Projector
from .utility.DLR import DLR
from .utility.Common import Common
from .utility.Fourier import Fourier
from .utility.Dyson import Dyson
from .utility.Mixing import Mixing
from .utility.Projection import Projection as PJ

class BLocDyn(object):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector):

        self.crystal = crystal
        self.dlr = dlr
        self.projector = projector

    def _as_dyn_dict(self, key = None, dyn : dict = None) -> dict:
        if dyn is not None:
            return dyn

        return {"1": np.zeros(len(self.dlr.nu), dtype=float).tolist()}

    def _write_json_pair(self, stem : str, iter : int, key, payload : dict) -> None:
        with open(f'{stem}.{iter}.{key}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))
        with open(f'{stem}.json', 'w') as outfile:
            json.dump(payload, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    def Inverse(self, matin : np.ndarray)-> np.ndarray:

        norb = matin.shape[0]
        ns = matin.shape[2]
        nft = matin.shape[4]

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
        tempmat = np.zeros((norb*ns,norb*ns),dtype=np.complex128)
        tempmat2 = np.zeros((norb*ns,norb*ns),dtype=np.complex128)

        for ift in range(nft):
            tempmat = self.crystal.OrbSpin2Composite(matin[...,ift])
            tempmat2 = Common.MatInv(tempmat)
            matout[...,ift] = self.crystal.Composite2OrbSpin(tempmat2)
        
        return matout

    # def Moment(self, bf : np.ndarray, oddzero : int, highzero : int) -> np.ndarray:

    #     norb = len(self.crystal.bind)
    #     ns = self.crystal.ns

    #     moment = np.zeros((norb,norb,ns,ns,3),dtype=np.complex128,order='F')
    #     high = np.zeros((norb,norb,ns,ns),dtype=np.complex128,order='F')
    #     moment, high = Fourier.BLocDynM(self.ft.nu,bf,oddzero,highzero)

    #     return moment,high
    
    def F2T(self,bf : np.ndarray) -> np.ndarray:

        norb = bf.shape[0]
        ns = bf.shape[2]
        nfreq = bf.shape[4]

        btau = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')

        bf_t = np.moveaxis(bf, -1, 0)
        batch = norb * norb * ns * ns
        bf_2d = np.ascontiguousarray(bf_t).reshape(nfreq, batch)

        from scipy.linalg import lu_solve
        G_xaa = lu_solve((self.dlr.dB.dlrmf2cf, self.dlr.dB.mf2cfpiv), bf_2d / self.dlr.beta)

        btau_2d = np.tensordot(self.dlr.dB.T_lx, G_xaa, axes=(1, 0))

        ntau = btau_2d.shape[0]
        btau = btau_2d.reshape(ntau, norb, norb, ns, ns)
        btau = np.moveaxis(btau, 0, -1)
        btau = np.asfortranarray(btau)

        return btau

    def T2F(self, btau : np.ndarray) -> np.ndarray:

        norb = btau.shape[0]
        ns = btau.shape[2]
        ntau = btau.shape[4]

        btau_t = np.moveaxis(btau, -1, 0)
        batch = norb * norb * ns * ns
        btau_2d = np.ascontiguousarray(btau_t).reshape(ntau, batch)

        from scipy.linalg import lu_solve
        fxx = lu_solve((self.dlr.dB.dlrit2cf, self.dlr.dB.it2cfpiv), btau_2d)
        bf_2d = self.dlr.beta * np.tensordot(
            self.dlr.dB.T_qx * self.dlr.dB.bosonic_corr_x[None, :], fxx, axes=(1, 0))
        
        nfreq = bf_2d.shape[0]
        bf = bf_2d.reshape(nfreq, norb, norb, ns, ns)
        bf = np.moveaxis(bf, 0, -1)  # (norb, norb, ns, ns, nrk, nfreq)
        bf = np.asfortranarray(bf)

        return bf

    def GaussianLinearBroad(self,x, y, w1, temperature, cutoff):

        norb = y.shape[0]
        ns = y.shape[2]
        nft = y.shape[3]

        ynew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')
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
                        for ks in range(ns):
                            for iorb in range(norb):
                                for jorb in range(norb):
                                    ynew[iorb,jorb,js,ks,cnt] = sum(dist*y[iorb,jorb,js,ks])/sum(dist)
                else:
                    ynew[...,cnt] = y[...,cnt]
            cnt += 1

        return ynew
    
    def Mixing(self,iter : int, mix : float, Bb : np.ndarray, Bold : np.ndarray):

        norb = Bb.shape[0]
        ns = Bb.shape[2]
        nft = Bb.shape[4]

        Bnew = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        if iter == 1:
            mix = 1.0
            Bold = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        Bnew = mix*Bb + (1-mix)*Bold

        return Bnew

    def _resolve_equiv_matrix(self, imp=None, key=None) -> np.ndarray:
        """Resolve an equivalent-orbital matrix from legacy/new impurity inputs."""
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

    def AverageByEquiv(self, equiv : np.ndarray, matin : np.ndarray, squeeze : bool = True) -> np.ndarray:
        """Average equivalent orbital classes and return array in one pass."""
        if matin.ndim == 3:
            if self.crystal.ns != 1:
                raise ValueError(
                    f"3D input is only allowed for ns=1, crystal ns={self.crystal.ns}"
                )
            matin5 = matin[:, :, np.newaxis, np.newaxis, :]
        elif matin.ndim == 5:
            matin5 = matin
        else:
            raise ValueError(f"matin must be 3D or 5D, got {matin.ndim}D")

        norb = matin5.shape[0]
        if matin5.shape[1] != norb:
            raise ValueError("matin first two dimensions must be square")
        if equiv.shape[0] != norb or equiv.shape[1] != norb:
            raise ValueError(
                f"equiv shape {equiv.shape} is incompatible with matin shape {matin5.shape}"
            )
        if matin5.shape[2] != self.crystal.ns or matin5.shape[3] != self.crystal.ns:
            raise ValueError(
                "spin dimension mismatch: "
                f"matin spin shape=({matin5.shape[2]}, {matin5.shape[3]}), "
                f"crystal ns={self.crystal.ns}"
            )

        matout = np.array(matin5, dtype=np.complex128, copy=True, order='F')
        nind = int(np.amax(equiv))
        if nind <= 0:
            raise ValueError("equiv labels must be positive integers")

        for ind in range(1, nind + 1):
            pos = Common.FindPositions(equiv, ind)
            if len(pos) == 0:
                continue
            for js in range(self.crystal.ns):
                for ks in range(self.crystal.ns):
                    avg = np.zeros(matin5.shape[4], dtype=np.complex128)
                    for ii, jj in pos:
                        avg += matin5[ii, jj, js, ks, :]
                    avg /= len(pos)
                    for ii, jj in pos:
                        matout[ii, jj, js, ks, :] = avg

        if squeeze and self.crystal.ns == 1:
            return matout[:, :, 0, 0, :]
        return matout

    def AverageImpurityByEquiv(self, imp=None, matimp : dict = None, squeeze : bool = True) -> dict:
        """Average equivalent orbital classes for all impurity problems at once."""
        if not isinstance(matimp, dict):
            raise TypeError("matimp must be dict keyed by impurity problem key")

        matout = {}
        for key, matin in matimp.items():
            equiv = self._resolve_equiv_matrix(imp=imp, key=key)
            matout[str(key)] = self.AverageByEquiv(equiv=equiv, matin=matin, squeeze=squeeze)

        return matout
    
    def Imp2Loc(self,matimp : np.ndarray)-> np.ndarray:

        norb = matimp.shape[0]
        ns = matimp.shape[2]
        nft = matimp.shape[3]

        nspace = 0
        for val in self.crystal.probspace.values():
            nspace += len(val)

        matloc = np.zeros((norb,norb,ns,ns,nft,nspace),dtype=np.complex128,order='F')

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

        matimp = np.zeros((norb,norb,ns,ns,nft,nprob),dtype=np.complex128,order='F')

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
                for ks in range(ns):
                    e = 0
                    for ii, jj in pos:
                        e += matin[ii,jj,js,ks]
                    e /=len(pos)
                    matdict[ind+1].append(e.tolist())
        
        return matdict
    
    def Dict2Arr(self,equiv : np.ndarray, matdict : np.ndarray) -> np.ndarray:

        norb = len(equiv)
        ns = self.crystal.ns
        nfreq = len(matdict["1"])                

        matout = np.zeros((norb,norb,ns,ns,nfreq),dtype=np.complex128,order='F')
        nind = np.amax(equiv)

        for js in range(ns):
            for ks in range(ns):
                for ind in range(nind):
                    pos = self.crystal.FindPositions(equiv,ind+1)
                    for ii, jj in pos:
                        matout[ii,jj,js,ks] = matdict[str(ind+1)]

        return matout

    def Dyson(self, mat1 : np.ndarray, mat2 : np.ndarray):

        norb = mat1.shape[0]
        ns = self.crystal.ns
        nft = self.ft.size

        matout = np.zeros((norb,norb,ns,ns,nft),dtype=np.complex128,order='F')

        matout = QAFort.dyson.blocdyn(mat1,mat2)

        return matout

    def Embedding(self, matin : np.ndarray):

        norb = len(self.crystal.bind)
        ns = self.crystal.ns
        nrk = len(self.crystal.kpoint)
        nft = self.ft.size
        nspace = self.crystal.bprojector.shape[3]

        matout = np.zeros((norb,norb,ns,ns,nrk,nft),dtype=np.complex128,order='F')

        for ispace in range(nspace):
            matout += QAFort.embedding.blocdyn(nrk,matin[...,ispace],self.crystal.bprojector[...,ispace])

        return matout

class BWeiss(BLocDyn):

    def __init__(self, crystal : Crystal, dlr : DLR, projector : Projector, vloc : VLoc, ploc, wloc):

        super().__init__(crystal, dlr, projector)

        self.vloc = vloc
        self.ploc = ploc
        self.wloc = wloc
# class PolLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, green, pol : object):
#         super().__init__(crystal, ft)
#         self.Cal()

#     def Cal(self):
#         pass

# class PolImp(BLocDyn): # read Polarizability from CTQMC

#     def __init__(self, crystal: Crystal, ft: FTGrid):
#         super().__init__(crystal, ft)

#         pass

# class WLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcLoc(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass

# class WcImp(BLocDyn):

#     def __init__(self, crystal: Crystal, ft: FTGrid, flocdyn: FLocDyn):
#         super().__init__(crystal, ft, flocdyn)

#         pass
