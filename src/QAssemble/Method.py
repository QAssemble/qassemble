from __future__ import annotations

import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys, time
import gc
import h5py
from .Crystal import Crystal
# from .FTGrid import FTGrid
from .utility.DLR import DLR
from .utility.Convergence import Convergence
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *
from .Projector import Projector
from .CTQMC import CTQMC


class GW(object):

    def __init__(self, Ginit : G, W : W, **kwargs):

        self.Ginit = Ginit
        self.W = W
        self.V = W.vbare
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'gw')
        self.iteration = kwargs.get('iteration', None)


    def __call__(self):

        crystal = self.Ginit.crystal
        dlr = self.Ginit.dlr


        sigh = SigH(crystal = crystal, occ = self.Ginit.occ, vbare = self.V.k, hdf5file = self.hdf5file, group = self.group)

        sigf = SigF(crystal = crystal, occr = self.Ginit.occr, vbare = self.V.r, hdf5file = self.hdf5file, group = self.group)

        siggwc = SigGWC(crystal = crystal, dlr = dlr, green = self.Ginit.rt, wlat = self.W.crt, hdf5file = self.hdf5file, group = self.group)

        p = P(crystal = crystal, dlr = dlr, green = self.Ginit.rt, hdf5file = self.hdf5file, group = self.group)


        if (self.iteration % 50 == 0) or (self.iteration == 1):
            sigh.Save(f'sigh.k.{self.iteration}')
            sigf.Save(f'sigf.k.{self.iteration}')
            siggwc.Save(f'siggwc.k.{self.iteration}')
            p.Save(f'p.kf.{self.iteration}')


        return sigh, sigf, siggwc, p

# Double Countaing Correction for GW+EDMFT
class GWLoc(object):

    def __init__(self, Ginit : GLoc, WLoc : WLoc, **kwargs):

        self.Ginit = Ginit
        self.WLoc = WLoc
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'gw')
        self.iteration = kwargs.get('iteration', None)


    def __call__(self):

        pass

class ImpuritySolver(object):

    def __init__(self, fweiss : FWeiss, bweiss : BWeiss, **kwargs):

        self.fweiss = fweiss
        self.bweiss = bweiss
        self.hdf5file = kwargs.get('hdf5file', None)
        self.group = kwargs.get('group', 'ctqmc')
        self.iteration = kwargs.get('iteration', None)

    def __call__(self):

        pass
