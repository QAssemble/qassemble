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
from matplotlib.legend import Legend

import DiagE

xx=np.zeros(100)
e0=np.zeros(100)
e1=np.zeros(100)
e2=np.zeros(100)
e3=np.zeros(100)
e4=np.zeros(100)
for ii in range(100):
    x=ii/100
    xx[ii]=x
    e0[ii]=DiagE.diagefourier.eulerpolynomial(x,0)
    e1[ii]=DiagE.diagefourier.eulerpolynomial(x,1)
    e2[ii]=DiagE.diagefourier.eulerpolynomial(x,2)
    e3[ii]=DiagE.diagefourier.eulerpolynomial(x,3)
    e4[ii]=DiagE.diagefourier.eulerpolynomial(x,4)


plt.close(1);plt.figure(1);plt.plot(xx, e0, xx, e1,xx, e2,xx, e3, xx, e4);plt.legend(['e0', 'e1', 'e2', 'e3', 'e4']);
plt.grid();
plt.show()

