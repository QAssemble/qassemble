from posixpath import normcase
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gc
import h5py
from .BPathDyn import *
from .BPathStc import *
from .FPathStc import *
from .FPathDyn import *
from .Crystal import Crystal
from .FTGrid import FTGrid
from .FLatDyn import FLatDyn
from .FLatStc import FLatStc
# from core.BLatDyn import BLatDyn
# from core.BLatStc import BLatStc

class PostFunction(object):

    def __init__(self, obj : object = None, hdf5file : str = 'glob.h5'):

        if (obj is not None):
            pass
        if (os.path.exists(hdf5file)):
            glob = h5py.File(hdf5file)
            ini = glob['input']
            tempcry = ini['Crystal']
            control = ini['Control']
            cry = {}
            kb = 8.6173303*10**-5
            for key in tempcry.keys():
                if (type(tempcry[key][()])==bytes):
                    cry[key] = str(tempcry[key][()],'utf8')
                else:
                    cry[key] = temp[key][()]
            for key in cry.keys():
                if key=='Basis':
                    cry[key] = eval(cry[key])
                elif key=='KGrid':
                    cry[key] = eval(cry[key])
                elif key=='RVec':
                    cry[key] = eval(cry[key])
                else:
                    cry[key] = cry[key]
            crystal = Crystal(cry)
            if ('T' in control)and('beta' not in control):
                T = control['T'][()]
                beta = 1/(T*kb)
            elif ('T' not in control)and('beta' in control):
                beta = control['beta'][()]
                T = 1/(beta*kb)
            cutoff = control.get('MatsubaraCutOff',50)
            ft = {}
            ft['T'] = T
            ft['beta'] = beta
            ft['cutoff'] = cutoff
            ftgrid = FTGrid(ft)
            glob.close()

            self.crystal = crystal
            self.ft = ftgrid

        self.fpathdyn = FPathDyn(crystal=crystal,ft=ftgrid,obj=obj,hdf5file=hdf5file)
        self.fpathstc = FPathStc(crystal=crystal,obj=obj,hdf5file=hdf5file)
        self.hdf5file = hdf5file
        self.obj = obj
