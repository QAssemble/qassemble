#!/bin/sh
#PBS -l nodes=1:ppn=12
#PBS -N temp
#PBS -q g2 // g2 : Gaussian, g1 : VASP


NPROCS=`wc -l < $PBS_NODEFILE`

hostname
#date >> /home/momichael98/Cho_Hyeon_Bin/result.log


cd $PBS_O_WORKDIR

#cp $PBS_NODEFILE nodefile

########################################
#For Gaussian with scratch
#export g09root=/GRAPE/Apps/GAUSSIAN
#export GAUSS_SCRDIR=/scratch
#export GV_DIR=/GRAPE/Apps/GAUSSIAN/gv
#source $g09root/g09/bsd/g09.profile
#g09 < input.com > output.log
########################################


#mpirun -genv I_MPI_DEBUG 5 -np $NPROCS /GRAPE/Apps/VASP/bin/5.3.5/NORMAL/vasp.5.3.5_31MAR2014_GRP7_NORMAL_VTST.x  > stdout.log
#mpirun -genv I_MPI_DEBUG 5 -np $NPROCS /GRAPE/Apps/VASP/bin/5.4.1/NORMAL/vasp_5.4.1_GRP7_NORMAL_p13082016.x  > stdout.log

source activate base

echo begin at: `date` >> logfile
#ipython Untitled.ipynb >> /home/momichael98/Cho_Hyeon_Bin/result.log
python GW_insulator.py >> /home/momichael98/temp/Fortran/DiagE/class/IPYTH_test/save_result1.log
python GW_graphene.py >> /home/momichael98/temp/Fortran/DiagE/class/IPYTH_test/save_result2.log
echo end at: `date` >> logfile
