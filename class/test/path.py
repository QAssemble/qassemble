import os
import sys
path = '/home/momichael98/temp/Fortran/DiagE/modules'

os.chdir(path)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(path))))
import DiagE

#import DiagE
