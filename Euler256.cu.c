// tatami_tiling_cuda_multi_gpu.cu

#include <iostream>
#include <cuda.h>
#include <vector>

const int TARGET_T = 200;        // Target T(s) value
const int MAX_DIMENSION = 50000; // Maximum dimension

// Adjust MAX_S based on your GPU memory capacity
const int64_t MAX_S = 100000000; // Maximum value of s

// Kernel function to compute T(s)
__global__ void compute_t_s_kernel(int64_t MAX_DIMENSION, int64_t MAX_S, int *T_s_count) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_threads = gridDim.x * blockDim.x;

    for (int64_t m = 2 + thread_id; m <= MAX_DIMENSION; m += total_threads) {
        for (int64_t n = m; n <= MAX_DIMENSION; ++n) {
            bool valid = false;

            // Tatami tiling conditions

            // Both dimensions are odd
            if (m % 2 == 1 && n % 2 == 1) {
                valid = true;
            }
            // Both dimensions are even, neither divisible by 4
            else if (m % 2 == 0 && n % 2 == 0 && m % 4 != 0 && n % 4 != 0) {
                valid = true;
            }
            // One dimension is odd, the other is even (even dimension not divisible by 4)
            else if (((m % 2 == 1 && n % 2 == 0) || (m % 2 == 0 && n % 2 == 1)) &&
                     ((m % 4 != 0 && m % 2 == 0) || (n % 4 != 0 && n % 2 == 0))) {
                valid = true;
            }

            if (valid) {
                int64_t s = m * n;

                if (s >= MAX_S) continue; // Avoid exceeding array bounds

                // Atomic addition to ensure thread safety
                atomicAdd(&T_s_count[s], 1);
            }
        }
    }
}

int main() {
    // Initialize CUDA devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2) {
        std::cerr << "At least two GPUs are required for this program." << std::endl;
        return 1;
    }

    // Use two GPUs
    int devices_to_use = 2;

    // For each GPU, we'll create a stream and launch the kernel
    std::vector<cudaStream_t> streams(devices_to_use);
    std::vector<int*> d_T_s_counts(devices_to_use);
    std::vector<int*> h_T_s_counts(devices_to_use);

    size_t size = MAX_S * sizeof(int);

    // Event timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Distribute workload among GPUs
    for (int device = 0; device < devices_to_use; ++device) {
        cudaSetDevice(device);

        // Allocate device memory
        cudaMalloc(&d_T_s_counts[device], size);
        cudaMemset(d_T_s_counts[device], 0, size);

        // Allocate host memory
        h_T_s_counts[device] = new int[MAX_S];
        memset(h_T_s_counts[device], 0, size);

        // Create streams
        cudaStreamCreate(&streams[device]);

        // Determine optimal block size and grid size
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy
        int gridSize;    // The actual grid size needed, based on input size

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compute_t_s_kernel, 0, 0);
        gridSize = (MAX_DIMENSION + blockSize - 1) / blockSize;

        // Launch the kernel on this device and stream
        compute_t_s_kernel<<<gridSize, blockSize, 0, streams[device]>>>(MAX_DIMENSION, MAX_S, d_T_s_counts[device]);
    }

    // Synchronize all streams
    for (int device = 0; device < devices_to_use; ++device) {
        cudaSetDevice(device);
        cudaStreamSynchronize(streams[device]);
    }

    // Copy results back to host and combine them
    std::vector<int> h_T_s_total(MAX_S, 0);

    for (int device = 0; device < devices_to_use; ++device) {
        cudaSetDevice(device);

        // Copy device memory to host
        cudaMemcpy(h_T_s_counts[device], d_T_s_counts[device], size, cudaMemcpyDeviceToHost);

        // Combine counts
        for (int64_t s = 0; s < MAX_S; ++s) {
            h_T_s_total[s] += h_T_s_counts[device][s];
        }

        // Clean up device memory and streams
        cudaFree(d_T_s_counts[device]);
        cudaStreamDestroy(streams[device]);

        // Free host memory
        delete[] h_T_s_counts[device];
    }

    // Event timing end
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Find the maximum T(s) and the smallest s for which T(s) == TARGET_T
    int max_T_s = 0;
    int64_t s_at_max_T_s = 0;
    bool found = false;
    int64_t min_s = 0;

    for (int64_t s = 0; s < MAX_S; ++s) {
        int count = h_T_s_total[s];
        if (count > max_T_s) {
            max_T_s = count;
            s_at_max_T_s = s;
        }
        if (count == TARGET_T) {
            if (!found || s < min_s) {
                min_s = s;
                found = true;
            }
        }
    }

    std::cout << "Maximum T(s) found: " << max_T_s << " at s = " << s_at_max_T_s << std::endl;

    if (found) {
        std::cout << "The smallest room size s for which T(s) = " << TARGET_T << " is: " << min_s << std::endl;
    } else {
        std::cout << "No result found within the maximum dimension of " << MAX_DIMENSION << "." << std::endl;
    }

    std::cout << "Total execution time: " << milliseconds / 1000.0 << " seconds." << std::endl;

    return 0;
}
