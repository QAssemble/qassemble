##### fortran
# FC = ifort
# INCLUDE = -I/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/include/fftw -I/home/sangkookchoi/usr/finufft-2.0.2/include
# FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -fp-model precise
# LDFLAGS = # -L
# LOADLIBES = -L/home/sangkookchoi/usr/finufft-2.0.2/lib -lfinufft -lfftw3xf_intel -mkl 


FC = ifort 
INCLUDE = -I${QAssemble}/finufft/include -I${MKLROOT}/include/fftw -I${MKLROOT}/include #-I/opt/homebrew/opt/fftw/include -I/opt/homebrew/opt/lapack/include
#FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -fp-model precise
FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -lmkl_rt#-DMPI -lblas -llapack -lpthread -lm -lfftw3 -llapack
LDFLAGS = # -L
LOADLIBES = ${QAssemble}/finufft/lib/libfinufft.so #-lfftw3 -llapack
#LOADLIBES = ../finufft/lib/libfinufft.so -lfftw3 -mkl
LAPACK_LIB = -L${MKLROOT}/lib/intel64#-llapack -lpthread -lm
FFTW = -I/opt/homebrew/opt/fftw/include
MKL = -I/opt/homebrew/opt/lapack/include
#LIBES = -L/home/momichael98/temp/Fortran/DiagE/modules/fftw3/fftw-3.3.10/lib -lfftw3 -mkl
#FFTW = -I/home/momichael98/temp/Fortran/DiagE/modules/fftw3/fftw-3.3.10/include
fortran2python = f2py -c --fcompiler=intelem --compiler=intelem  $(LOADLIBES) $(INCLUDE)

##### finufft
CXX=icpc
CC=icc
Finufft_CFLAGS= -O3 -xHost -I${MKLROOT}/include/fftw -lmkl_rt
LIBS = -L${MKLROOT}/lib/intel64/ -Wl, -rpath=${MKLROOT}/lib/intel64
LDFLAGS += $(LIBS)
LIBSFFT = -lmkl_rt

