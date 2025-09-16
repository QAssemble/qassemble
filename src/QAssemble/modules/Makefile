include ../../../arch.mk

.SUFFIXES: .f90 .o

OBJ = \
  Common.o\
  Dyson.o\
  Projection.o\
  Embedding.o\
  Fourier.o\
  Bare.o\

all: libqassemble.so

libqassemble.so: $(OBJ)
	$(FC) -shared $(FFLAGS) -L/opt/homebrew/opt/fftw/lib -lfftw3 -L/Users/moseongjun/Desktop/Code/DiagE/finufft/lib -lfinufft -o $@ $^

python:
	$(fortran2python) -m QAFort Common.f90 Dyson.f90 Projection.f90 Embedding.f90 Fourier.f90 Bare.f90 $<

new:
	make clean
	make all

.f90.o:
	$(FC) -c -fPIC $(FFLAGS) -L$(LAPACK_LIB) $(INCLUDE) -L$(LOADLIBES) -L/opt/homebrew/opt/fftw/lib -lfftw3 -L/Users/moseongjun/Desktop/Code/DiagE/finufft/lib -lfinufft $<

clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf *.exe
