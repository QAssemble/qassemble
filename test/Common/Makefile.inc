include ../../arch.mk

.SUFFIXES: .f90 .o

OBJ = \
  Common.o

all: $(OBJ)

new:
	make clean
	make all
.f90.o:
	$(FC) -c $(FFLAG) $(INCLUDE) $(LOADLIBES) $<
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	

