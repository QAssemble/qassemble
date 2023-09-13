import numpy as np
import sys
path = "/home/momichael98/temp/Fortran/DiagE/modules"
sys.path.append(path)
import DiagE

F_test = np.ones(10,dtype=complex,order='F')
DiagE.common.fftw3_1d(F_test,-1)
F_test2 = np.ones(10,dtype=complex,order='F')
DiagE.common.fftw3_1d(F_test2,-1)
F_test2=np.flip(F_test2)
print(F_test)
print('-----')
print(F_test2)
