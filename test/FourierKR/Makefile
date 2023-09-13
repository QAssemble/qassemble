include ../../barch.mk

.SUFFIXES: .f90 .o

TARGET = TestFourierKR

OBJ = \
  TestFourierKR.o

OBJCommon = \
  ../../modules/Fourier.o \
  ../../modules/Common.o \

all: $(OBJ) 
	$(FC) $(FFLAGS) $(MKL) $(FFTW) $(LIBES) $(INCLUDE) $(LOADLIBES) $(OBJ) $(OBJCommon) -I../../modules -o $(TARGET)	
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


