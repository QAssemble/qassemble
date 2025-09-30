# makefile overrides to use Intel ICC compiler & libiomp5. double prec only.
# (fixed the fatal mistake of leaving -lgomp in the compile line! 6/4/20)
include ../arch.mk

CXX:=$(CXX)
CC:=$(CC)
FC:=$(FC)

# we want to start fresh, ignore the GCC flags...
CFLAGS += $(Finufft_CFLAGS) #-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/fftw/include 
# CFLAGS += -lsvml    % fails (gives nans & no faster)

CXXFLAGS = $(CFLAGS)
FFLAGS   = $(CFLAGS)
LIBS := $(LIBS) #-L/opt/homebrew/opt/fftw/lib -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp 
LDFLAGS += $(LIBS)
LIBSFFT = $(LIBSFFT) #-lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp

# It is crucial to *replace* -fopenmp & -lgomp with Intel's equiv
# (crucial in the sense that linking libiomp5 *and* libgomp gives weird
# segfaults, corruption in parallel blocks, or works fine, frustratingly):
OMPFLAGS = -fopenmp
OMPLIBS =

