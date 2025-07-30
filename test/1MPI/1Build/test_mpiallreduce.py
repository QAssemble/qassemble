import numpy as np
from mpi4py import MPI

# 1. Initialize MPI Environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# 2. Define Global Problem Parameters
# NOTE: ntau is the full size here, not a local slice.
# The shape is the same on all processes.
norb = 4
ntau = 128
nmag = 2
global_shape = (norb, norb, ntau, nmag)

# 3. Create a Local Version of the Full Array on Each Process
# In a real case, this would be the result of a simulation on this rank.
# We'll fill it with a unique value (rank + 1) to make the sum easy to verify.
gtau_local_copy = np.full(global_shape, float(rank + 1), dtype=np.float64)

# 4. Perform the MPI_Allreduce
# Create a buffer to hold the result of the reduction.
gtau_sum = np.empty(global_shape, dtype=np.float64)

# Perform the element-wise sum of gtau_local_copy from all processes.
# The result, gtau_sum, will be identical on all processes.
comm.Allreduce(
    sendbuf=gtau_local_copy,
    recvbuf=gtau_sum,
    op=MPI.SUM
)

# 5. Post-process (e.g., Calculate the Average)
# The sum is now on every process, so this is a local operation.
gtau_average = gtau_sum / nproc

# 6. Verification
# We can check the result on any rank.
# The sum of numbers from 1 to nproc is nproc * (nproc + 1) / 2.
# The average is (nproc + 1) / 2.
expected_sum_val = nproc * (nproc + 1) / 2
expected_avg_val = (nproc + 1) / 2

# Check if a random element in the result has the expected value.
# Using a tolerance for floating point comparison.
sum_is_correct = np.allclose(gtau_sum[0, 0, 0, 0], expected_sum_val)
avg_is_correct = np.allclose(gtau_average[0, 0, 0, 0], expected_avg_val)

if rank == 0:
    print(f"Running on {nproc} processes.")
    print(f"Shape of array on each process: {global_shape}")
    print("-" * 30)
    print(f"Value of a test element in gtau_sum: {gtau_sum[0, 0, 0, 0]}")
    print(f"Expected sum value: {expected_sum_val}")
    print(f"Sum verification: {'SUCCESS' if sum_is_correct else 'FAIL'}")
    print("-" * 30)
    print(f"Value of a test element in gtau_average: {gtau_average[0, 0, 0, 0]}")
    print(f"Expected average value: {expected_avg_val}")
    print(f"Average verification: {'SUCCESS' if avg_is_correct else 'FAIL'}")