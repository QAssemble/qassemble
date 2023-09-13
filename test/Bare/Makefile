include ../../barch.mk

.SUFFIXES: .f90 .o

TARGET = TestBare  

OBJ = \
  TestBare.o

OBJCommon = \
  ../../modules/Bare.o \
  ../../modules/Common.o \
  ../../modules/Fourier.o \

all: $(OBJ)
	$(FC) $(FFLAGS) $(MKL) $(FFTW) $(LIBES) $(INCLUDE) $(LOADLIBES) $(OBJ) $(OBJCommon) -I../../modules -o $(TARGET)
new:
	make clean
	make all
.f90.o:
	$(FC) -c $(FFLAGS) $(INCLUDE) $(LOADLIBES) -I../../modules $<
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf $(TARGET)


