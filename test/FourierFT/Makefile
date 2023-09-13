include ../../barch.mk

.SUFFIXES: .f90 .o

TARGET = TestFourierFT

OBJ = \
  TestFourierFT.o

OBJCommon = \
  ../../modules/Fourier.o \
  ../../modules/Common.o \
#  ../Common/Common.o

all: $(OBJ)   
	$(FC)  $(FFLAGS) $(FFTW) $(MKL) $(LIBES) $(INCLUDE) $(LOADLIBES) $(OBJ) $(OBJCommon) -I../../modules -o $(TARGET)
.f90.o: 
	$(FC) -c $(FFLAGS) $(INCLUDE) -L$(LOADLIBES) -I../../modules $<

new:
	make clean
	make all 
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf $(TARGET)


