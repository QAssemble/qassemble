import numpy as np
from Common import Common
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

"""
FFT.py
======

This module defines the `FFT` class, which serves as a wrapper for performing parallel Fast Fourier Transforms (FFT)
across multiple MPI ranks using the `mpi4py_fft` library. It manages the setup, data distribution (slicing),
and execution of forward and backward transforms.
"""

class FFT():
    """
    A class to handle parallel FFT operations.

    This class initializes a parallel FFT object (`PFFT`) and manages the distributed arrays required for
    computations. It also calculates and stores the local slice indices and shapes for every rank in the
    MPI communicator, facilitating easy access to data distribution information.

    Attributes:
        fft (PFFT): The parallel FFT object configured for the given shape and communicator.
        arr (distArray): Distributed array container used in transforms.
        arrT (distArray): Distributed array container used in transforms.
        slicef (dict): Dictionary mapping all ranks to their respective forward slices.
        sliceb (dict): Dictionary mapping all ranks to their respective backward slices.
        localshapef (dict): Dictionary mapping all ranks to their respective forward shapes.
        localshapeb (dict): Dictionary mapping all ranks to their respective backward shapes.
    """

    def __init__(self, comm : MPI.COMM_WORLD, shape : list):
        """
        Initialize the FFT class.

        Args:
            comm (MPI.Comm): The MPI communicator.
            shape (list): The global shape of the array to be transformed.
                          If shape[2] == 1, a specific grid configuration is used.
        """

        if (shape[2] == 1):
            self.fft = PFFT(comm, shape=shape, axes = (0, 1, 2), dtype = np.complex128, grid = (-1,))
        else:
            self.fft = PFFT(comm, shape=shape, axes = (0, 1, 2), dtype = np.complex128)

        self.arr = newDistArray(self.fft, forward_output=True)
        self.arrT = newDistArray(self.fft, forward_output=False)
        localslicef_temp = self.fft.local_slice(forward_output=True)
        localsliceb_temp = self.fft.local_slice(forward_output=False)

        self.localslicef = [(s.start, s.stop) for s in localslicef_temp]
        self.localsliceb = [(s.start, s.stop) for s in localsliceb_temp]
        self.localshapef = tuple(s[1] - s[0] for s in self.localslicef)
        self.localshapeb = tuple(s[1] - s[0] for s in self.localsliceb)

        all_slicef = comm.allgather(self.localslicef)
        all_sliceb = comm.allgather(self.localsliceb)
        all_localshapef = comm.allgather(self.localshapef)
        all_localshapeb = comm.allgather(self.localshapeb)

        self.slicef = {rank: all_slicef[rank] for rank in range(comm.Get_size())}
        self.sliceb = {rank: all_sliceb[rank] for rank in range(comm.Get_size())}
        self.localshapef = {rank: all_localshapef[rank] for rank in range(comm.Get_size())}
        self.localshapeb = {rank: all_localshapeb[rank] for rank in range(comm.Get_size())}

    def Forward(self, matin):
        """
        Perform the Forward FFT.

        Details:
            Copies the input data `matin` into `self.arrT`, then performs a forward FFT
            storing the result in `self.arr`.

        Args:
            matin (ndarray): Input array suitable for the local slice of the backward layout.

        Returns:
            distArray: The result of the forward FFT (stored in `self.arr`).
        """

        val = self.arrT.copy()
        result = self.arr.copy()
        val[:] = matin
        result = self.fft.forward(val, result, normalize=False)

        return result

    def Backward(self, matin):
        """
        Perform the Backward FFT.

        Details:
            Copies the input data `matin` into `self.arr`, then performs a backward FFT
            storing the result in `self.arrT`.

        Args:
            matin (ndarray): Input array suitable for the local slice of the forward layout.

        Returns:
            distArray: The result of the backward FFT (stored in `self.arrT`).
        """

        val = self.arr.copy()
        result = self.arrT.copy()
        val[:] = matin
        result = self.fft.backward(val, result, normalize = False)

        return result