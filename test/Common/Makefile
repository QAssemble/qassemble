include ../../barch.mk

.SUFFIXES: .f90 .o

TARGET = TestCommon

OBJ = \
  TestCommon.o

OBJCommon = \
  ../../modules/Common.o \

all: $(OBJ)
	$(FC) $(FFLAGS) $(MKL) $(FFTW) $(LIBES) $(OBJ) $(OBJCommon) -I../../modules -o $(TARGET)
new:
	make clean
	make all
.f90.o:
	$(FC) -c $(FFLAGS) $(MKL) $(FFTW) $(LIBES) -I../../modules $<
clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf *.mod
	rm -rf $(TARGET)


