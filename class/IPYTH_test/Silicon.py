import numpy as np
import sys
path = '/home/momichael98/temp/Fortran/DiagE/class'
sys.path.append(path)
from ClassDiagE_new import Crystal as cr
from ClassDiagE_new import FHamiltonian as fh
from ClassDiagE_new import BHamiltonian as bh
from ClassDiagE_new import FT_grid as ft
from ClassDiagE_new import Method
path = '/home/momichael98/temp/Fortran/DiagE/modules'
sys.path.append(path)
import DiagE
import matplotlib
import matplotlib.pyplot as plt