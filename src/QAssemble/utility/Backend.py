import os

def CheckMPI():

    ismpi = False
    # for OpenMPI:
    if os.environ.get('OMPI_COMM_WORLD_RANK'):
        ismpi = True
    # for MPICH and intel based MPI:
    elif os.environ.get('PMI_RANK'):
        ismpi = True
    # for PMIx (used by srun/Slurm with PMIx support):
    elif os.environ.get('PMIX_RANK'):
        ismpi = True
    elif os.environ.get('CRAY_MPICH_VERSION'):
        ismpi = True
    # to force the MPI init manually
    elif os.environ.get('TRIQS_FORCE_MPI_INIT'):
        ismpi = True
    else:
        print('Warning: could not identify MPI environment!')
        print('The calculation proceeds using the serial version.')

    return ismpi



