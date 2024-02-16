include arch.mk

default : all

all : finufft modules com_ctqmc
clean : clean_finufft clean_diage clean_ctqmc

finufft :
	cd finufft && $(MAKE) fortran && cd ../

modules : 
	cd modules && $(MAKE) python && cd ../
com_ctqmc:
	cd ComCTQMC && $(MAKE) cpu && cd ../

clean_finufft:
	cd finufft && $(MAKE) clean && cd ../

clean_diage:
	cd modules && $(MAKE) clean && cd ../

clean_ctqmc:
	cd ComCTQMC && $(MAKE) clean && cd ../
