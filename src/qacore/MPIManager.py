from mpi4py import MPI
import sys
import numpy as np


class MPIManager(object):

    def __init__(self, comm : MPI.COMM_WORLD):

        print("Parallelization with MPI Start")
        self.comm = comm 

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.submatrixk = []
        self.submatrixw = []

    def Split(self, nprock : int = None, nprocw : int = None, nk : int = None, nw : int = None):
        
        if (nprock * nprocw != self.size):
            if (self.rank == 0):
                print(f"Error: nprock*nprocw = {nprock * nprocw}, but MPI world size = {self.size}")
                raise ValueError("nprock*nprocw must equal MPI world size")
            sys.exit(1)
        else:
            print(f"nprock*nprocw = {nprock * nprocw}, MPI world size = {self.size}")
        # Require both axes list and array
        kidx = self.rank // nprocw
        widx = self.rank % nprocw

        kper = nk // nprock
        wper = nw // nprocw

        k0, k1 = kidx * kper, (kidx + 1) * kper
        w0, w1 = widx * wper, (widx + 1) * wper
        # Create sub-communicators
        commk = self.comm.Split(color=kidx, key=widx)
        commw = self.comm.Split(color=widx, key=kidx)

        self.submatrixk.append((k0, k1))

        self.submatrixw.append((w0, w1))

        print(
        f"[rank {self.rank:>2}] coords=(k={kidx},w={widx}), "
        f"k_range=[{k0}:{k1}), w_range=[{w0}:{w1})  "
        f"sizes: comm_k={commk.Get_size()}, comm_w={commw.Get_size()}"
    )

        return commk, commw

    # def Split(self, dimensions : list = None, A : np.ndarray = None):
        
    #     # npart = len(dimensions)

    #     # if (npart == 2):
    #     #     n1 = A.shape[dimensions[0]]
    #     #     n2 = A.shape[dimensions[1]]

    #     # elif (npart == 3):
    #     #     n1 = A.shape[dimensions[0]]
    #     #     n2 = A.shape[dimensions[1]]
    #     #     n3 = A.shape[dimensions[2]]
    #     # elif (npart == 4):
    #     #     n1 = A.shape[dimensions[0]]
    #     #     n2 = A.shape[dimensions[1]]
    #     #     n3 = A.shape[dimensions[2]]
    #     #     n4 = A.shape[dimensions[3]]
    #     # elif (npart == 5):
    #     #     n1 = A.shape[dimensions[0]]
    #     #     n2 = A.shape[dimensions[1]]
    #     #     n3 = A.shape[dimensions[2]]
    #     #     n4 = A.shape[dimensions[3]]
    #     #     n5 = A.shape[dimensions[4]]
    #     # elif (npart == 6):
    #     #     n1 = A.shape[dimensions[0]]
    #     #     n2 = A.shape[dimensions[1]]
    #     #     n3 = A.shape[dimensions[2]]
    #     #     n4 = A.shape[dimensions[3]]
    #     #     n5 = A.shape[dimensions[4]]
    #     #     n6 = A.shape[dimensions[5]]

    #     # Split array A across MPI processes along specified axes.
    #     if dimensions is None or A is None:
    #         raise ValueError("Both 'dimensions' and 'A' must be provided")
    #     dims = dimensions
    #     k = len(dims)
    #     # Determine process grid shape: default to equal splits matching array lengths
    #     procs = [A.shape[axis] for axis in dims]
    #     if np.prod(procs) != self.size:
    #         raise ValueError(f"Product of process counts {procs} must equal MPI size {self.size}")
    #     # Create a Cartesian communicator for splitting
    #     cart_comm = self.comm.Create_cart(procs, periods=[False]*k)
    #     coords = cart_comm.Get_coords(self.rank)
    #     # Build slice objects for local chunk
    #     slices = [slice(None)] * A.ndim
    #     for i, axis in enumerate(dims):
    #         total = A.shape[axis]
    #         count = procs[i]
    #         block = total // count
    #         rem = total % count
    #         # Distribute the remainder among the first 'rem' ranks
    #         if coords[i] < rem:
    #             start = coords[i] * (block + 1)
    #             length = block + 1
    #         else:
    #             start = coords[i] * block + rem
    #             length = block
    #         slices[axis] = slice(start, start + length)
    #     local_A = A[tuple(slices)]
    #     return local_A, cart_comm


    
# class FLatDynMPI(MPIManager):
    
#     def __init__(self, comm : MPI.COMM_WORLD):
#         super().__init__(comm)
#         self.comm = comm
#         self.rank = self.comm.Get_rank()
#         self.size = self.comm.Get_size()

#     def Split(self, nprock : int = None, nprocw : int = None, A : np.ndarray = None):
        
#         if (nprock * nprocw != self.size):
#             if (self.rank == 0):
#                 print(f"Error: nprock*nprocw = {nprock * nprocw}, but MPI world size = {self.size}")
#                 raise ValueError("nprock*nprocw must equal MPI world size")
#             sys.exit(1)

#         nk = A.shape[3]
#         nw = A.shape[4]

#         commk, commw = super().Split(nprock, nprocw, nk, nw)

#         return commk, commw
    


# class BLatDynMPI(MPIManager):
    
#     def __init__(self, comm : MPI.COMM_WORLD):
#         super().__init__(comm)
#         self.comm = comm
#         self.rank = self.comm.Get_rank()
#         self.size = self.comm.Get_size()

#     def Split(self, A = None):
#         """
#         Block distribution of a linear collection A across MPI ranks.
#         A may be a Python sequence (supports __len__ and __getitem__) or a singly-linked list with 'next' attribute.
#         Returns (local_chunk, [comm]).
#         """
#         if A is None:
#             raise ValueError("A must be provided")
#         # Sequence support
#         try:
#             n = len(A)
#             block = n // self.size
#             rem = n % self.size
#             if self.rank < rem:
#                 start = self.rank * (block + 1)
#                 length = block + 1
#             else:
#                 start = self.rank * block + rem
#                 length = block
#             local = A[start:start + length]
#             return local, [self.comm]
#         except Exception:
#             pass
#         # Linked-list support
#         if not hasattr(A, 'next'):
#             raise TypeError("A must be a sequence or linked list with 'next' attribute")
#         # compute total length
#         n = 0
#         curr = A
#         while curr is not None:
#             n += 1
#             curr = getattr(curr, 'next', None)
#         block = n // self.size
#         rem = n % self.size
#         if self.rank < rem:
#             start = self.rank * (block + 1)
#             length = block + 1
#         else:
#             start = self.rank * block + rem
#             length = block
#         # advance to start
#         idx = 0
#         curr = A
#         while idx < start and curr is not None:
#             curr = getattr(curr, 'next', None)
#             idx += 1
#         new_head = curr
#         new_tail = curr
#         # traverse required length and sever
#         cnt = 1
#         while cnt < length and new_tail is not None:
#             new_tail = getattr(new_tail, 'next', None)
#             cnt += 1
#         if new_tail is not None:
#             next_node = getattr(new_tail, 'next', None)
#             new_tail.next = None
#         return new_head, [self.comm]

class FLatDynMPI(MPIManager):

    def __init__(self, comm : MPI.COMM_WORLD = None):
        super().__init__(comm)
        # self.comm = comm
        # self.rank = super().rank
        # self.size = super().size

    def Split(self, nprock : int = None, nprocw : int = None, A : np.ndarray = None):
        
        # if (nprock * nprocw != self.size):
        #     if (self.rank == 0):
        #         print(f"Error: nprock*nprocw = {nprock * nprocw}, but MPI world size = {self.size}")
        #         raise ValueError("nprock*nprocw must equal MPI world size")
        #     sys.exit(1)

        nk = A.shape[3]
        nw = A.shape[4]

        commk, commw = super().Split(nprock, nprocw, nk, nw)

        return commk, commw