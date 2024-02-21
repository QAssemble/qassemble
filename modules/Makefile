include ../arch.mk

.SUFFIXES: .f90 .o

OBJ = \
  Common.o\
  Dyson.o\
  Projection.o\
  Embedding.o\
  Fourier.o\
  Bare.o\

all: $(OBJ)

python:
	$(fortran2python) -m DiagE Common.f90 Dyson.f90 Projection.f90 Embedding.f90 Fourier.f90 Bare.f90

new:
	make clean
	make all
.f90.o:
	$(FC) -c $(FFLAGS) $(MKL) $(FFTW) $(LIBES) $(INCLUDE) $(LOADLIBES) $<
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf *.exe
