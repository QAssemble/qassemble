include ../../arch.mk

.SUFFIXES: .f90 .o

TARGET = TestProjection

OBJ = \
  TestProjection.o

OBJCommon = \
  ../../modules/Projection.o \
  ../../modules/Common.o \
  ../Common/Common.o

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


