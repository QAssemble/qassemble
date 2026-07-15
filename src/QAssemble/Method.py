from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *
from .CTQMC import CTQMC

@dataclass
class HFResult:
    sigh: SigH
    sigf: SigF

    def __iter__(self):
        yield self.sigh
        yield self.sigf

@dataclass
class HFLocResult:
    sigh: SigHLoc
    sigf: SigFLoc

    def __iter__(self):
        yield self.sigh
        yield self.sigf

@dataclass
class GWResult:
    siggwc: SigGWC
    pol: P

    def __iter__(self):
        yield self.siggwc
        yield self.pol

@dataclass
class GWLocResult:
    siggwc: SigGWCLoc
    pol: PLoc

    def __iter__(self):
        yield self.siggwc
        yield self.pol

@dataclass
class ImpurityActionResult:
    ctqmc: CTQMC
    key: str
    gimp: GImp
    sighimp: SigHImp
    sigfimp: SigFImp
    sigimp: SigCImp
    chi: Chi
    pimp: PImp
    wimp: WImp
    diagnostics: dict

class HF(object):

    def __init__(self, occ : np.ndarray, occr : np.ndarray, v : V, **kwargs):

        self.occ = occ
        self.occr = occr
        self.v = v
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'hf')
        self.iteration = kwargs.get('iteration', None)
        self.mix = kwargs.get('mix', None)
        self.mixing_method = kwargs.get('mixing_method', 'pulay')
        self.npulay = int(kwargs.get('npulay', 5))

    def __call__(self):
        t0 = time.perf_counter()

        crystal = self.v.crystal

        sigh = SigH(
            crystal=crystal,
            occ=self.occ,
            vbare=self.v.k,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        sigf = SigF(
            crystal=crystal,
            occr=self.occr,
            vbare=self.v.r,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        if self.iteration is not None and ((self.iteration % 50 == 0) or (self.iteration == 1)):
            sigh.Save('sigh.k')
            sigf.Save('sigf.k')

        if self.mix is not None:
            sigh.Mixing(
                mix=self.mix,
                method=self.mixing_method,
                npulay=self.npulay,
            )
            sigf.Mixing(
                mix=self.mix,
                method=self.mixing_method,
                npulay=self.npulay,
            )

        self.elapsed = time.perf_counter() - t0
        return HFResult(sigh=sigh, sigf=sigf)


class HFLoc(object):

    def __init__(self, gloc : GLoc = None, vloc : VLoc = None, **kwargs):

        self.gloc = gloc if gloc is not None else kwargs.get('gloc', None)
        self.vloc = vloc if vloc is not None else kwargs.get('vloc', None)
        self.key = kwargs.get('key', getattr(self.vloc, 'key', None))
        if self.gloc is None:
            raise ValueError(f"{self.__class__.__name__} requires GLoc")
        if self.vloc is None:
            raise ValueError(f"{self.__class__.__name__} requires VLoc")
        if self.key is None:
            raise ValueError(f"{self.__class__.__name__} requires an impurity problem key")
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'hf')
        self.iteration = kwargs.get('iteration', None)

    def __call__(self):

        crystal = self.gloc.crystal
        projector = self.gloc.projector
        if getattr(self.gloc, "key", self.key) != self.key:
            raise ValueError(
                f"{self.__class__.__name__} key '{self.key}' does not match "
                f"GLoc key '{self.gloc.key}'"
            )
        occ = self.gloc.occ
        if self.key not in self.vloc.vproj:
            self.vloc.BuildProjection(projector)
        vloc = self.vloc.vproj[self.key]

        sigh = SigHLoc(
            crystal=crystal,
            projector=projector,
            key=self.key,
            occ=occ,
            vloc=vloc,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        sigf = SigFLoc(
            crystal=crystal,
            projector=projector,
            key=self.key,
            occ=occ,
            vloc=vloc,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        if (self.iteration % 50 == 0) or (self.iteration == 1):
            sigh.Save('sighloc')
            sigf.Save('sigfloc')

        return HFLocResult(sigh=sigh, sigf=sigf)


class GW(object):

    def __init__(self, g : G = None, w : W = None, **kwargs):

        self.g = g if g is not None else kwargs.get('g', kwargs.get('Ginit', None))
        self.w = w if w is not None else kwargs.get('w', kwargs.get('W', None))
        if self.g is None:
            raise ValueError(f"{self.__class__.__name__} requires G")
        if self.w is None:
            raise ValueError(f"{self.__class__.__name__} requires W")
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'gw')
        self.iteration = kwargs.get('iteration', None)
        self.mix = kwargs.get('mix', None)
        self.mixing_method = kwargs.get('mixing_method', 'pulay')
        self.npulay = int(kwargs.get('npulay', 5))


    def __call__(self):
        t0 = time.perf_counter()

        crystal = self.g.crystal
        dlr = self.g.dlr

        siggwc = SigGWC(
            crystal=crystal,
            dlr=dlr,
            green=self.g.rt,
            wlat=self.w.crt,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        p = P(
            crystal=crystal,
            dlr=dlr,
            green=self.g.rt,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )


        if self.iteration is not None and ((self.iteration % 50 == 0) or (self.iteration == 1)):
            siggwc.Save('siggwc.k')
            p.Save('p.kf')

        if self.mix is not None:
            siggwc.Mixing(
                mix=self.mix,
                method=self.mixing_method,
                npulay=self.npulay,
            )
            p.Mixing(
                mix=self.mix,
                method=self.mixing_method,
                npulay=self.npulay,
            )

        self.elapsed = time.perf_counter() - t0
        return GWResult(siggwc=siggwc, pol=p)

# Double Countaing Correction for GW+EDMFT
class GWLoc(object):

    def __init__(self, gloc : GLoc, wloc : WLoc, **kwargs):

        self.gloc = gloc
        self.wloc = wloc
        self.key = kwargs.get('key', getattr(wloc, 'key', None))
        if self.gloc is None:
            raise ValueError(f"{self.__class__.__name__} requires GLoc")
        if self.wloc is None:
            raise ValueError(f"{self.__class__.__name__} requires WLoc")
        if self.key is None:
            raise ValueError(f"{self.__class__.__name__} requires an impurity problem key")
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'gw')
        self.iteration = kwargs.get('iteration', None)


    def __call__(self):

        crystal = self.gloc.crystal
        dlr = self.gloc.dlr
        projector = self.gloc.projector
        if getattr(self.gloc, "key", self.key) != self.key:
            raise ValueError(
                f"{self.__class__.__name__} key '{self.key}' does not match "
                f"GLoc key '{self.gloc.key}'"
            )
        green = self.gloc.t

        siggwc = SigGWCLoc(
            crystal=crystal,
            dlr=dlr,
            projector=projector,
            key=self.key,
            green=green,
            wloc=self.wloc.ct,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        p = PLoc(
            crystal=crystal,
            dlr=dlr,
            projector=projector,
            key=self.key,
            gloc=green,
            hdf5file=self.hdf5file,
            group=self.group,
            iteration=self.iteration,
        )

        if (self.iteration % 50 == 0) or (self.iteration == 1):
            siggwc.Save('siggwcloc.f')
            p.Save('ploc.f')

        return GWLocResult(siggwc=siggwc, pol=p)

class ImpurityAction(object):

    def __init__(self, fweiss : FWeiss, bweiss : BWeiss, **kwargs):

        self.fweiss = fweiss
        self.bweiss = bweiss
        self.key = kwargs.get('key', None)
        if self.key is None:
            self.key = getattr(fweiss, 'key', getattr(bweiss, 'key', None))
        self.control = kwargs.get('control', None)
        self.hdf5file = kwargs.get('hdf5file', getattr(fweiss, 'hdf5file', None))
        self.group = kwargs.get('group', getattr(fweiss, 'group', None))
        if self.group is None:
            self.group = 'ctqmc'
        self.iteration = kwargs.get('iteration', 1)

    def __call__(self, iteration : int = None):

        iter = self.iteration if iteration is None else iteration
        if iter is None:
            raise ValueError("ImpurityAction requires an iteration number")
        if self.key is None:
            raise ValueError("ImpurityAction requires an impurity problem key")

        ctqmc = CTQMC(
            dlr=self.fweiss.dlr,
            fweiss=self.fweiss,
            bweiss=self.bweiss,
            key=self.key,
            control=self.control,
            hdf5file=self.hdf5file,
            group=self.group,
        )
        ctqmc.PreProcessing(iter=iter)
        ctqmc.Run(iter=iter)
        ctqmc.PostProcessing(iter=iter)

        self.key = getattr(ctqmc, 'key', self.key)

        return ImpurityActionResult(
            ctqmc=ctqmc,
            key=self.key,
            gimp=getattr(ctqmc, 'gimp', None),
            sighimp=getattr(ctqmc, 'sighimp', None),
            sigfimp=getattr(ctqmc, 'sigfimp', None),
            sigimp=getattr(ctqmc, 'sigimp', None),
            chi=getattr(ctqmc, 'chi', None),
            pimp=getattr(ctqmc, 'pimp', None),
            wimp=getattr(ctqmc, 'wimp', None),
            diagnostics=dict(getattr(ctqmc, 'diagnostics', {})),
        )
