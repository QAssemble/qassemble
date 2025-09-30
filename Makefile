include arch.mk

default : all

all: qa_finufft qa_modules

qa_finufft:
	cd finufft && $(MAKE) fortran && cd ../
qa_modules:
	cd src/qacore/modules && $(MAKE) python && cd ../../..

clean: clean_finufft clean_modules

clean_finufft:
	cd finufft && $(MAKE) clean && cd ../
clean_modules:
	cd src/qacore/modules && $(MAKE) clean && cd ../../..
