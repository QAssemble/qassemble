##### fortran
# FC = ifort
# INCLUDE = -I/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/include/fftw -I/home/sangkookchoi/usr/finufft-2.0.2/include
# FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -fp-model precise
# LDFLAGS = # -L
# LOADLIBES = -L/home/sangkookchoi/usr/finufft-2.0.2/lib -lfinufft -lfftw3xf_intel -mkl 


FC = ifort
INCLUDE = -I/home/sangkookchoi/usr/finufft-2.0.2/include
FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -fp-model precise
LDFLAGS = # -L
LOADLIBES = /home/sangkookchoi/usr/finufft-2.0.2/lib/libfinufft.so -lfftw3xf_intel -mkl
fortran2python = f2py -c --fcompiler=intelem --compiler=intelem $(LOADLIBES) $(INCLUDE)



