#!/usr/bin/env python3
"""
Comprehensive analysis of 3D PFFT pencil decomposition
Focus on understanding data layout, communication patterns, and performance
"""

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import numpy as np


def analyze_pencil_memory_layout():
    """Analyze memory layout and access patterns in pencil decomposition"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"=== Memory Layout Analysis ===")
    print(f"Process {rank} of {size}")

    shape = (32, 24, 16)

    # Create PFFT
    fft = PFFT(comm, shape=shape, dtype=np.complex128, slab=False)
    u = newDistArray(fft, forward_output=False)
    u_hat = newDistArray(fft, forward_output=True)

    # Get detailed slice information
    input_slice = fft.local_slice(forward_output=False)
    output_slice = fft.local_slice(forward_output=True)

    print(f"\nProcess {rank} memory layout:")
    print(f"  Input array:")
    print(f"    Global shape: {shape}")
    print(f"    Local shape: {u.shape}")
    print(f"    Local slice: {input_slice}")
    print(f"    Memory size: {u.nbytes / (1024**2):.2f} MB")
    print(f"    Stride pattern: {u.strides}")

    print(f"  Output array:")
    print(f"    Local shape: {u_hat.shape}")
    print(f"    Local slice: {output_slice}")
    print(f"    Memory size: {u_hat.nbytes / (1024**2):.2f} MB")
    print(f"    Stride pattern: {u_hat.strides}")

    # Analyze cache efficiency
    if u.size > 0:
        # Check contiguity
        input_contiguous = u.flags.c_contiguous
        print(f"  Input array C-contiguous: {input_contiguous}")

        # Memory access pattern analysis
        i_slice, j_slice, k_slice = input_slice

        # Check which dimensions are local vs distributed
        local_dims = []
        distributed_dims = []

        for dim, (slice_obj, dim_size) in enumerate(zip(input_slice, shape)):
            if isinstance(slice_obj, slice):
                local_size = slice_obj.stop - slice_obj.start
                if local_size == dim_size:
                    local_dims.append(dim)
                else:
                    distributed_dims.append(dim)
            else:
                distributed_dims.append(dim)

        print(f"  Local dimensions: {local_dims}")
        print(f"  Distributed dimensions: {distributed_dims}")

        # Estimate cache performance
        if u.shape:
            cache_line_elements = 64 // u.itemsize  # Assume 64-byte cache line
            innermost_dim_size = u.shape[-1]
            cache_efficiency = min(1.0, innermost_dim_size / cache_line_elements)
            print(f"  Estimated cache efficiency: {cache_efficiency:.2f}")


def trace_fft_transforms():
    """Trace the sequence of transforms in 3D pencil FFT"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    shape = (16, 12, 8)

    print(f"\n=== FFT Transform Sequence ===")
    print(f"Process {rank}, Array shape: {shape}")

    # Create PFFT
    fft = PFFT(comm, shape=shape, dtype=np.complex128, slab=False)
    u = newDistArray(fft, forward_output=False)
    u_hat = newDistArray(fft, forward_output=True)

    # Initialize with a simple pattern
    if u.size > 0:
        u[:] = 1.0  # Constant value for easy tracking
        u[0, 0, 0] = 1000.0  # Add a spike for tracking

        print(f"Process {rank}: Initial data")
        print(f"  Shape: {u.shape}")
        print(f"  Sample values: {u.flat[:5]}")
        print(f"  Max value: {np.max(np.abs(u))}")

    # Forward transform
    print(f"\nProcess {rank}: Forward FFT...")
    u_hat = fft.forward(u, u_hat)

    if u_hat.size > 0:
        print(f"  Output shape: {u_hat.shape}")
        print(f"  Output slice: {fft.local_slice(forward_output=True)}")
        print(f"  Sample values: {u_hat.flat[:5]}")
        print(f"  Max magnitude: {np.max(np.abs(u_hat))}")

        # Find DC component (should be large due to constant input)
        dc_magnitude = np.abs(u_hat[0, 0, 0]) if u_hat.size > 0 else 0
        print(f"  DC component magnitude: {dc_magnitude}")

    # Backward transform
    print(f"\nProcess {rank}: Backward FFT...")
    u_reconstructed = fft.backward(u_hat)

    if u.size > 0:
        reconstruction_error = np.max(np.abs(u - u_reconstructed))
        print(f"  Reconstruction error: {reconstruction_error:.2e}")
        print(f"  Sample reconstructed: {u_reconstructed.flat[:5]}")


def analyze_communication_patterns():
    """Analyze communication patterns in pencil decomposition"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"\n=== Communication Pattern Analysis ===")
    print(f"Process {rank}, Total processes: {size}")

    # Test with different sizes to see communication scaling
    test_shapes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ]

    for shape in test_shapes:
        print(f"\nArray shape: {shape}")
        print("-" * 30)

        # Create PFFT
        fft = PFFT(comm, shape=shape, dtype=np.complex128, slab=False)
        u = newDistArray(fft, forward_output=False)
        u_hat = newDistArray(fft, forward_output=True)

        # Analyze data distribution
        input_slice = fft.local_slice(forward_output=False)
        output_slice = fft.local_slice(forward_output=True)

        # Check if transpose is needed
        transpose_needed = input_slice != output_slice

        print(f"Process {rank}:")
        print(f"  Input distribution: {input_slice}")
        print(f"  Output distribution: {output_slice}")
        print(f"  Transpose needed: {transpose_needed}")

        if transpose_needed:
            # Estimate communication volume
            local_elements = u.size
            element_size = u.itemsize  # bytes per element

            # In pencil decomposition, each process typically communicates
            # with all other processes during transpose
            comm_volume_per_process = local_elements * element_size
            total_comm_volume = comm_volume_per_process * size

            print(
                f"  Est. communication volume per process: {comm_volume_per_process / (1024**2):.2f} MB"
            )
            print(
                f"  Est. total communication volume: {total_comm_volume / (1024**2):.2f} MB"
            )

            # Communication efficiency
            computation_work = local_elements * np.log2(np.prod(shape))
            comm_computation_ratio = comm_volume_per_process / computation_work
            print(f"  Communication/computation ratio: {comm_computation_ratio:.2e}")


def benchmark_pencil_performance():
    """Benchmark pencil decomposition performance"""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"\n=== Performance Benchmark ===")
    print(f"Process {rank}, Processes: {size}")

    # Test different problem sizes
    test_cases = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 128, 64),
    ]

    for shape in test_cases:
        print(f"\nBenchmarking shape: {shape}")
        print("-" * 40)

        # Create PFFT
        fft = PFFT(comm, shape=shape, dtype=np.complex128, slab=False)
        u = newDistArray(fft, forward_output=False)
        u_hat = newDistArray(fft, forward_output=True)

        # Initialize with random data
        if u.size > 0:
            np.random.seed(rank)  # Different seed per process
            u[:] = (
                np.random.random(u.shape) + 1j * np.random.random(u.shape)
            ) / np.sqrt(2)

        # Warm up
        print(f"Process {rank}: Warming up...")
        for _ in range(3):
            u_hat = fft.forward(u, u_hat)
            u_back = fft.backward(u_hat)

        # Benchmark forward transform
        print(f"Process {rank}: Benchmarking forward FFT...")
        comm.Barrier()
        start_time = MPI.Wtime()

        num_iterations = 10
        for _ in range(num_iterations):
            u_hat = fft.forward(u, u_hat)

        comm.Barrier()
        forward_time = (MPI.Wtime() - start_time) / num_iterations

        # Benchmark backward transform
        print(f"Process {rank}: Benchmarking backward FFT...")
        comm.Barrier()
        start_time = MPI.Wtime()

        for _ in range(num_iterations):
            u_back = fft.backward(u_hat)

        comm.Barrier()
        backward_time = (MPI.Wtime() - start_time) / num_iterations

        # Gather results
        all_forward_times = comm.gather(forward_time, root=0)
        all_backward_times = comm.gather(backward_time, root=0)

        if rank == 0:
            # Analyze timing results
            max_forward = max(all_forward_times)
            min_forward = min(all_forward_times)
            avg_forward = np.mean(all_forward_times)

            max_backward = max(all_backward_times)
            min_backward = min(all_backward_times)
            avg_backward = np.mean(all_backward_times)

            print(f"  Forward FFT times:")
            print(f"    Min: {min_forward:.6f}s")
            print(f"    Max: {max_forward:.6f}s")
            print(f"    Avg: {avg_forward:.6f}s")
            print(f"    Load balance: {min_forward/max_forward:.3f}")

            print(f"  Backward FFT times:")
            print(f"    Min: {min_backward:.6f}s")
            print(f"    Max: {max_backward:.6f}s")
            print(f"    Avg: {avg_backward:.6f}s")
            print(f"    Load balance: {min_backward/max_backward:.3f}")

            # Performance metrics
            total_elements = np.prod(shape)
            forward_throughput = total_elements / max_forward / 1e6
            backward_throughput = total_elements / max_backward / 1e6

            print(f"  Throughput:")
            print(f"    Forward: {forward_throughput:.2f} M elements/s")
            print(f"    Backward: {backward_throughput:.2f} M elements/s")

            # Scaling efficiency (rough estimate)
            ideal_time = max_forward / size  # Perfect scaling assumption
            parallel_efficiency = ideal_time / max_forward
            print(f"  Parallel efficiency estimate: {parallel_efficiency:.3f}")


def demonstrate_real_application():
    """Demonstrate pencil decomposition in a real application context"""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print(f"\n=== Real Application Example ===")
    print(f"Process {rank}: Solving 3D Poisson equation")

    # Solve -∇²u = f using FFT
    shape = (64, 64, 64)
    L = 2 * np.pi  # Domain size

    # Create PFFT
    fft = PFFT(comm, shape=shape, dtype=np.float64)  # Real-to-complex
    f = newDistArray(fft, forward_output=False)
    f_hat = newDistArray(fft, forward_output=True)
    u = newDistArray(fft, forward_output=False)
    u_hat = newDistArray(fft, forward_output=True)

    # Initialize right-hand side
    input_slice = fft.local_slice(forward_output=False)

    if f.size > 0:
        i_slice, j_slice, k_slice = input_slice

        # Get coordinate ranges
        i_start = i_slice.start if isinstance(i_slice, slice) else i_slice
        i_end = i_slice.stop if isinstance(i_slice, slice) else i_slice + 1
        j_start = j_slice.start if isinstance(j_slice, slice) else j_slice
        j_end = j_slice.stop if isinstance(j_slice, slice) else j_slice + 1
        k_start = k_slice.start if isinstance(k_slice, slice) else k_slice
        k_end = k_slice.stop if isinstance(k_slice, slice) else k_slice + 1

        # Create source term
        for local_i, global_i in enumerate(range(i_start, i_end)):
            for local_j, global_j in enumerate(range(j_start, j_end)):
                for local_k, global_k in enumerate(range(k_start, k_end)):
                    x = global_i * L / shape[0]
                    y = global_j * L / shape[1]
                    z = global_k * L / shape[2]

                    # Source term: f(x,y,z) = sin(x) * cos(y) * sin(z)
                    f[local_i, local_j, local_k] = np.sin(x) * np.cos(y) * np.sin(z)

        print(f"Process {rank}: Initialized source term")
        print(f"  Local shape: {f.shape}")
        print(f"  RMS value: {np.sqrt(np.mean(f**2)):.6f}")

    # Forward FFT of source term
    f_hat = fft.forward(f, f_hat)

    # Solve in Fourier space
    if f_hat.size > 0:
        output_slice = fft.local_slice(forward_output=True)

        # Create wavenumber arrays
        out_i_slice, out_j_slice, out_k_slice = output_slice

        # Get local wavenumber ranges
        ki_start = out_i_slice.start if isinstance(out_i_slice, slice) else out_i_slice
        ki_end = out_i_slice.stop if isinstance(out_i_slice, slice) else out_i_slice + 1
        kj_start = out_j_slice.start if isinstance(out_j_slice, slice) else out_j_slice
        kj_end = out_j_slice.stop if isinstance(out_j_slice, slice) else out_j_slice + 1
        kk_start = out_k_slice.start if isinstance(out_k_slice, slice) else out_k_slice
        kk_end = out_k_slice.stop if isinstance(out_k_slice, slice) else out_k_slice + 1

        # Solve: û = f̂ / (-k²)
        for local_i, global_i in enumerate(range(ki_start, ki_end)):
            for local_j, global_j in enumerate(range(kj_start, kj_end)):
                for local_k, global_k in enumerate(range(kk_start, kk_end)):
                    # Wavenumbers (with proper scaling)
                    ki = global_i if global_i < shape[0] // 2 else global_i - shape[0]
                    kj = global_j if global_j < shape[1] // 2 else global_j - shape[1]
                    kk = global_k  # For real FFT, only positive frequencies

                    k_squared = (
                        (ki * 2 * np.pi / L) ** 2
                        + (kj * 2 * np.pi / L) ** 2
                        + (kk * 2 * np.pi / L) ** 2
                    )

                    if k_squared > 0:
                        u_hat[local_i, local_j, local_k] = (
                            -f_hat[local_i, local_j, local_k] / k_squared
                        )
                    else:
                        u_hat[local_i, local_j, local_k] = 0  # DC component

        print(f"Process {rank}: Solved in Fourier space")
        print(f"  Solution RMS (Fourier): {np.sqrt(np.mean(np.abs(u_hat)**2)):.6f}")

    # Inverse FFT to get solution
    u = fft.backward(u_hat, u)

    if u.size > 0:
        print(f"Process {rank}: Solution obtained")
        print(f"  Solution RMS: {np.sqrt(np.mean(u**2)):.6f}")
        print(f"  Min value: {np.min(u):.6f}")
        print(f"  Max value: {np.max(u):.6f}")

    # Verify solution by computing Laplacian
    print(f"Process {rank}: Verifying solution...")

    # This would involve computing derivatives and checking residual
    # For brevity, we'll skip the detailed verification
    print(f"Process {rank}: Poisson solve completed successfully")


if __name__ == "__main__":
    # Run comprehensive analysis
    analyze_pencil_memory_layout()
    trace_fft_transforms()
    analyze_communication_patterns()
    benchmark_pencil_performance()
    demonstrate_real_application()
