import string as string
import matplotlib as mat
import re as re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import matplotlib.font_manager as fm
from collections import OrderedDict
import json, os, shutil
import itertools
from scipy.fftpack import fftn, ifftn

# I need projector, noninteracting_term

class Ctest():
    def __init__(self,x, y, z=None, a=None):
        self.x=x
        self.y=y
        if (z is None):
            self.z=str(x+y)
        else:
            self.z=z
        if (a is None):
            self.a=np.arange(x+y)
        else:
            self.a=a
        print(self.a)

        
