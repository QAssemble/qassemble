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
from .utility.Projection import Projection as PJ

logger = logging.getLogger("QAssemble")

class FLocStc(object):

    def __init__(self,crystal : Crystal, projector : Projector):

        self.crystal = crystal
        self.projector = projector

    def ResolveProblemKey(self, key):
        if self.projector is None:
            raise ValueError("projector is required to resolve problem key")
        pkey = key if key in self.projector.fprojector else str(key)
        if pkey not in self.projector.fprojector:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        return pkey

    def Inverse(self,mat : np.ndarray):

        norb = mat.shape[0]
        ns = mat.shape[2]

        matinv = np.zeros((norb,norb,ns),dtype=np.complex128,order='F')

        for js in range(ns):
            matinv[:,:,js] = Common.MatInv(mat[:, :, js])

        return matinv

    def Mixing(self,iter : int, mix : float, Fb : np.ndarray, Fold : np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Object-local single-array mixing is no longer supported. "
            "Use utility.Mixing with HDF5-backed quantities."
        )

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
    
    def Projection(self, matin : np.ndarray, key):
        if self.projector is None:
            raise ValueError("projector is required for Projection")

        if matin.ndim != 4:
            raise ValueError(f"matin must be 4D, got {matin.ndim}D")

        pkey = self.ResolveProblemKey(key)
        return PJ.FLatStc(matin, self.projector.fprojector[pkey])
    
    
class EImp(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, key, hamtb : np.ndarray, mu : float, sigh : np.ndarray = None, sigf : np.ndarray = None, hloc : np.ndarray = None, floc : np.ndarray = None, hdf5file : str = None, group : str = None):

        super().__init__(crystal, projector)
        print("Enter the EImp class")
        self.key = self.ResolveProblemKey(key)
        self.hamtb = hamtb
        self.mu = mu
        self.ham = None
        self.sig = None
        self.sigh = None
        self.sigf = None
        self.hloc = hloc
        self.floc = floc
        self.e = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        print("Non Interacting Hamiltonian : ", self.hamtb[:, :, 0, 0])
        print("Chemical Potential : ", self.mu)
        print("Hartree Self Energy : ", sigh[:, :, 0] if sigh is not None else None)
        print("Fock Self Energy : ", sigf[:, :, 0] if sigf is not None else None)
        print("Double Counting Hartree : ", hloc[:, :, 0] if hloc is not None else None)
        print("Double Counting Fock : ", floc[:, :, 0] if floc is not None else None)

        tempmat = np.zeros_like(hamtb, dtype=np.complex128, order='F')

        for ik in range(hamtb.shape[3]):
            for js in range(hamtb.shape[2]):
                tempmat[...,js,ik] = hamtb[...,js,ik] - mu*np.eye(hamtb.shape[0], dtype=np.complex128)

        
        self.sigh = self._resolve_static_self_energy("sigh", sigh)
        self.sigf = self._resolve_static_self_energy("sigf", sigf)
        self.sig = self.sigh[..., np.newaxis] + self.sigf[..., np.newaxis]
        self.ham = tempmat + self.sig
        
        self.Cal()

        print("Exit the EImp class")

    def _resolve_static_self_energy(self, name : str, sigma : np.ndarray) -> np.ndarray:
        key = self.key
        norbc = self.projector.fprojector[key].shape[1]
        ns = self.crystal.ns

        if sigma is None:
            return np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

        sigma = np.asarray(sigma, dtype=np.complex128)

        if sigma.ndim == 3:
            expected = (norbc, norbc, ns)
            if sigma.shape != expected:
                raise ValueError(
                    f"{name} local static self-energy shape {sigma.shape} "
                    f"is incompatible with expected {expected}"
                )
            return np.asfortranarray(sigma)

        if sigma.ndim == 4:
            if sigma.shape[:2] != self.hamtb.shape[:2]:
                raise ValueError(
                    f"{name} lattice static self-energy orbital shape "
                    f"{sigma.shape[:2]} is incompatible with hamtb shape "
                    f"{self.hamtb.shape[:2]}"
                )
            if sigma.shape[2] != ns:
                raise ValueError(
                    f"{name} spin dimension {sigma.shape[2]} is incompatible "
                    f"with crystal ns={ns}"
                )
            if sigma.shape[3] != self.hamtb.shape[3]:
                raise ValueError(
                    f"{name} k dimension {sigma.shape[3]} is incompatible "
                    f"with hamtb k dimension {self.hamtb.shape[3]}"
                )
            return np.asfortranarray(self.Projection(sigma, key))

        raise ValueError(
            f"{name} must be a 3D local or 4D lattice static self-energy, "
            f"got {sigma.ndim}D"
        )

    def Cal(self):

        e = self.Projection(self.ham, self.key)
        # e = e + self.sig

        
        if (self.hloc is not None) and (self.floc is not None):
            self.e = e - self.hloc - self.floc
        else:
            self.e = e
        print(f"[EImp.Cal] key={self.key}, e[0,0,0]={self.e[0,0,0]}, mu={self.mu}")
        logger.info("Impurity Level : %s", self.e)
        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            eimp = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(eimp, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(eimp, fn, self.e, dtype=complex)

        return None

    def ToCTQMC(self, key = None, Eimp : np.ndarray = None):
        key = self.ResolveProblemKey(self.key if key is None else key)
        ns = self.crystal.ns
        norbc = self.projector.fprojector[key].shape[1]

        if Eimp is None:
            Eimp = self.e

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
        
        # return A_final,ctqmc_mu
        return A_final, ctqmc_mu.real
            
            
            
class SigHLoc(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, key, occ : np.ndarray = None, vloc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.key = self.ResolveProblemKey(key)
        self.occ = occ
        self.vloc = vloc
        self.hloc = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    
    def Cal(self):

        key = self.key
        proj = self.projector.fprojector[key]
        norbc = proj.shape[1]
        ns = proj.shape[2]
        v = self.vloc
        norb = v.shape[0]

        h = np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

        if ns != 1:

            for ind1 in range(norb * ns):
                nn1 = [0] * 2
                ind1, [iorb, js] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind1, nn1)

                iorbc1, iorbc2 = self.projector.ProbBorb2FPair(key, iorb)

                for ind2 in range(norb * ns):
                    nn2 = [0] * 2
                    ind2, [jorb, ks] = Common.Indexing(norb*ns, 2, [norb, ns], 0, ind2, nn2)

                    iorbc3, iorbc4 = self.projector.ProbBorb2FPair(key, jorb)

                    h[iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[iorbc4, iorbc3, ks])
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

                    h[iorbc1, iorbc2, js] += (v[iorb, jorb, js, ks] * self.occ[iorbc4, iorbc3, ks]) * C

        self.hloc = h

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            sighloc = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(sighloc, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(sighloc, fn, self.hloc, dtype=complex)

        return None


class SigHImp(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, key, gimp = None, occ : np.ndarray = None, vloc = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.key = self.ResolveProblemKey(key)
        self.gimp = gimp
        self.occ = occ
        self.vloc = vloc
        self.s = None
        self.h = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _occupation_from_gimp(self):

        if self.gimp is None or self.gimp.t is None:
            raise ValueError("SigHImp requires occ or a GImp object with tau data")

        return -self.gimp.t[:, :, :, -1].copy()

    def _local_interaction(self):

        if self.vloc is None:
            raise ValueError("SigHImp requires vloc or utilde interaction data")

        v = self.vloc.vloc if hasattr(self.vloc, "vloc") else self.vloc
        if isinstance(v, dict):
            v = v[self.key]
        v = np.asarray(v, dtype=np.complex128)
        bproj = self.projector.bprojector[self.key]
        norbb = bproj.shape[1]

        if v.ndim == 6:
            iprob = int(self.key) - 1
            v = v[..., 0, iprob]
        elif v.ndim == 5:
            v = v[..., 0]
        elif v.ndim != 4:
            raise ValueError(f"interaction must be 4D, 5D, or 6D, got {v.ndim}D")

        if v.shape[0] != norbb:
            v = PJ.BLocStc(v, bproj)

        return np.asfortranarray(v)

    def Cal(self):

        occ = self.occ if self.occ is not None else self._occupation_from_gimp()
        v = self._local_interaction()

        norbb = v.shape[0]
        ns = v.shape[2]
        norbc = occ.shape[0]
        h = np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

        if ns != 1:
            for ind1 in range(norbb * ns):
                nn1 = [0] * 2
                _, [iorb, js] = Common.Indexing(norbb * ns, 2, [norbb, ns], 0, ind1, nn1)
                iorbc1, iorbc2 = self.projector.ProbBorb2FPair(self.key, iorb)

                for ind2 in range(norbb * ns):
                    nn2 = [0] * 2
                    _, [jorb, ks] = Common.Indexing(norbb * ns, 2, [norbb, ns], 0, ind2, nn2)
                    iorbc3, iorbc4 = self.projector.ProbBorb2FPair(self.key, jorb)
                    h[iorbc1, iorbc2, js] += v[iorb, jorb, js, ks] * occ[iorbc4, iorbc3, ks]
        else:
            C = 1 if self.crystal.soc else 2
            for ind1 in range(norbb * ns):
                nn1 = [0] * 2
                _, [iorb, js] = Common.Indexing(norbb * ns, 2, [norbb, ns], 0, ind1, nn1)
                iorbc1, iorbc2 = self.projector.ProbBorb2FPair(self.key, iorb)

                for ind2 in range(norbb * ns):
                    nn2 = [0] * 2
                    _, [jorb, ks] = Common.Indexing(norbb * ns, 2, [norbb, ns], 0, ind2, nn2)
                    iorbc3, iorbc4 = self.projector.ProbBorb2FPair(self.key, jorb)
                    h[iorbc1, iorbc2, js] += v[iorb, jorb, js, ks] * occ[iorbc4, iorbc3, ks] * C

        self.occ = occ
        self.h = h
        self.s = h

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            sighimp = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(sighimp, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(sighimp, fn, self.s, dtype=complex)

        return None

class SigFLoc(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, key, occ : np.ndarray = None, vloc : np.ndarray = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.key = self.ResolveProblemKey(key)
        self.occ = occ
        self.vloc = vloc
        self.floc = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__

        self.Cal()

    def Cal(self):

        key = self.key
        proj = self.projector.fprojector[key]
        norbc = proj.shape[1]
        ns = proj.shape[2]
        v = self.vloc
        norb = v.shape[0]

        f = np.zeros((norbc, norbc, ns), dtype=np.complex128, order='F')

        for ind1 in range(norb * ns):
            nn1 = [0] * 2
            ind1, [iorb, js] = Common.Indexing(norb * ns, 2, [norb, ns], 0, ind1, nn1)

            iorbc1, iorbc4 = self.projector.ProbBorb2FPair(key, iorb)

            for ind2 in range(norb * ns):
                nn2 = [0] * 2
                ind2, [jorb, ks] = Common.Indexing(norb * ns, 2, [norb, ns], 0, ind2, nn2)

                iorbc3, iorbc2 = self.projector.ProbBorb2FPair(key, jorb)

                if js == ks:
                    f[iorbc1, iorbc2, js] += (
                        -self.occ[iorbc4, iorbc3, js] * v[iorb, jorb, js, ks]
                    )

        self.floc = f

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            sigfloc = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(sigfloc, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(sigfloc, fn, self.floc, dtype=complex)

        return None


class SigFImp(FLocStc):

    def __init__(self, crystal : Crystal, projector : Projector, key, sigma, sigh = None, hdf5file : str = 'glob.h5', group : str = None):

        super().__init__(crystal, projector)

        self.key = self.ResolveProblemKey(key)
        self.sigma_in = sigma
        self.sigh = sigh
        self.hf = None
        self.s = None
        self.hdf5file = hdf5file
        self.group = group
        self.subgroup = self.__class__.__name__
        self.Cal()

    def _read_ctqmc_sigma_hf(self, sigma : dict) -> dict:

        matdict = {}
        for sigma_key, val in sigma.items():
            if isinstance(val, dict) and "moments" in val:
                matdict[sigma_key] = complex(val["moments"][0])
            else:
                arr = np.asarray(val, dtype=np.complex128)
                matdict[sigma_key] = complex(arr.flat[0])

        return matdict

    def Cal(self):

        if isinstance(self.sigma_in, dict):
            equiv = np.asarray(self.projector.equiv[self.key], dtype=int)
            sigma = self._read_ctqmc_sigma_hf(self.sigma_in)
            self.hf = self.ReadDict(equiv, sigma)
        else:
            self.hf = np.asfortranarray(self.sigma_in, dtype=np.complex128)

        if self.sigh is None:
            self.s = np.array(self.hf, dtype=np.complex128, copy=True, order='F')
        else:
            sigh = self.sigh.s if hasattr(self.sigh, "s") else self.sigh
            self.s = np.asfortranarray(self.hf - sigh)

        return None

    def Save(self, fn: str, obj : np.ndarray = None):

        with h5py.File(self.hdf5file,'a') as file:
            sigfimp = Common.HDF5Subgroup(file, self.group, self.subgroup)
            if obj is not None:
                Common.HDF5CreateDataset(sigfimp, fn, obj, dtype=complex)
            else:
                Common.HDF5CreateDataset(sigfimp, fn, self.s, dtype=complex)

        return None





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
