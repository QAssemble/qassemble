##### fortran
# FC = ifort
# INCLUDE = -I/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl/include/fftw -I/home/sangkookchoi/usr/finufft-2.0.2/include
# FFLAGS = -debug -g -CB -check bounds -traceback -check uninit -fp-model precise
# LDFLAGS = # -L
# LOADLIBES = -L/home/sangkookchoi/usr/finufft-2.0.2/lib -lfinufft -lfftw3xf_intel -mkl 

FC = gfortran
INCLUDE = -I${QAssemble}/finufft/include -I/opt/homebrew/opt/fftw/include -I/opt/homebrew/opt/lapack/include
FFLAGS = -DMPI -lblas -llapack -lpthread -lm -lfftw3 -llapack
LDFLAGS = # -L
LOADLIBES = ${QAssemble}/finufft/lib/libfinufft.so #-lfftw3 -llapack
LAPACK_LIB = /opt/homebrew/opt/lapack/lib#-llapack -lpthread -lm
FFTW = -I/opt/homebrew/opt/fftw/include
MKL = -I/opt/homebrew/opt/lapack/include
fortran2python = /Users/moseongjun/.pyenv/versions/3.9.16/bin/f2py -c --fcompiler=gfortran --compiler=unix  $(LOADLIBES) $(LAPACK_LIB)/liblapack.dylib $(INCLUDE)

##### finufft
CXX=g++-14
CC=gcc-14
Finufft_CFLAGS=-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/fftw/include
LIBS = -L/opt/homebrew/opt/fftw/lib -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp
LDFLAGS += $(LIBS)
LIBSFFT = -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp

