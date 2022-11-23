include ../arch.mk

.SUFFIXES: .f90 .o

OBJ = \
  Common.o\
  Dyson.o\
  Bare.o\
  Projection.o\
  Embedding.o\
  Fourier.o\
  MatsubaraBasis.o\

all: $(OBJ)

new:
	make clean
	make all
.f90.o: 
	$(FC) -c $(FFLAGS) $(INCLUDE) $(LOADLIBES) $<
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf *.exe
