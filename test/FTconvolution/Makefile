include ../../barch.mk

.SUFFIXES: .f90 .o

# TARGET = FourierFtoT_Finu_chebyshev_boson
TARGET = TestFourierF2T_Fermion

# OBJ = \
  # FourierFtoT_Finu_chebyshev_boson.o
OBJ = \
  TestFourierF2T_Fermion.o

OBJCommon = \
  ../../modules/MatsubaraBasis.o
  # ../../modules/Fourier.o \
  # ../../modules/Common.o \



all: $(OBJ)
	$(FC)  $(FFLAGS) $(INCLUDE) $(LOADLIBES) $(OBJ) $(OBJCommon) -I../../modules -o $(TARGET)
.f90.o:
	$(FC) -c $(FFLAGS) $(INCLUDE) $(LOADLIBES) -I../../modules $<

new:
	make clean
	make all 
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf $(TARGET)


