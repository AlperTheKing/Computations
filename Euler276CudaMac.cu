#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <mutex>

// Define maximum perimeter
const unsigned long long int MAX_PERIMETER = 10000000;

// Define threshold for printing triangles (e.g., every 1,000,000th triangle)
const unsigned long long int PRINT_THRESHOLD = 1000000000; // Adjust to 1000000000 as needed

// Mutex for synchronized output (host-side)
std::mutex print_mutex;

// Device function to compute GCD of two numbers
__device__ unsigned long long int device_gcd(unsigned long long int a, unsigned long long int b) {
    while (b != 0) {
        unsigned long long int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Device function to compute GCD of three numbers
__device__ unsigned long long int device_gcd_three(unsigned long long int a, unsigned long long int b, unsigned long long int c) {
    return device_gcd(a, device_gcd(b, c));
}

// Kernel to count primitive triangles
__global__ void count_triangles_kernel(unsigned long long int start_p, unsigned long long int end_p, unsigned long long int* d_count, unsigned long long int* d_next_threshold) {
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int p = start_p + idx;

    if (p >= end_p) return;

    unsigned long long int local_count = 0;

    for (unsigned long long int a = 1; a <= p / 3; ++a) {
        for (unsigned long long int b = a; b <= (p - a) / 2; ++b) {
            unsigned long long int c = p - a - b;
            if (a + b > c) { // Triangle inequality
                if (device_gcd_three(a, b, c) == 1) { // Primitive check
                    local_count++;
                }
            }
        }
    }

    // Atomic addition to global count
    unsigned long long int new_count = atomicAdd(d_count, local_count) + local_count;

    // Check if new_count >= next_threshold
    unsigned long long int threshold = *d_next_threshold;
    if (new_count >= threshold) {
        // Attempt to increment next_threshold atomically
        if (atomicCAS(d_next_threshold, threshold, threshold + PRINT_THRESHOLD) == threshold) {
            // This thread successfully updated the threshold
            // Note: Printing specific (a, b, c) is non-trivial without additional tracking
            printf("Reached %llu primitive triangles.\n", new_count);
        }
    }
}

// Function to determine optimal block and grid sizes using occupancy API
void determine_block_grid_sizes(int device_id, unsigned long long int per_device_perimeters, int& optimal_block_size, int& optimal_grid_size) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    int minGridSize;
    int blockSize_temp;

    // Determine the block size and grid size to maximize occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_temp, count_triangles_kernel, 0, 0);
    optimal_block_size = blockSize_temp;
    optimal_grid_size = (per_device_perimeters + optimal_block_size - 1) / optimal_block_size;
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Step 1: Get number of CUDA-capable devices
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    checkCudaError(err, "Getting device count");

    if (device_count < 2) {
        std::cerr << "Error: This program requires at least two CUDA-capable GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    // Step 2: Divide perimeters among devices
    std::vector<unsigned long long int> per_device_perimeters(device_count, 0);
    unsigned long long int per_device = MAX_PERIMETER / device_count;
    for (int i = 0; i < device_count; ++i) {
        per_device_perimeters[i] = per_device;
    }
    // Assign remaining perimeters to the last device
    per_device_perimeters[device_count - 1] += MAX_PERIMETER % device_count;

    // Step 3: Create streams and allocate memory
    std::vector<cudaStream_t> streams(device_count);
    std::vector<unsigned long long int*> d_counts(device_count, nullptr);
    std::vector<unsigned long long int*> d_next_thresholds(device_count, nullptr);
    std::vector<unsigned long long int> h_counts(device_count, 0);
    std::vector<unsigned long long int> h_next_thresholds(device_count, PRINT_THRESHOLD);

    for (int i = 0; i < device_count; ++i) {
        // Set device
        cudaSetDevice(i);

        // Create stream
        err = cudaStreamCreate(&streams[i]);
        checkCudaError(err, "Creating stream");

        // Allocate device memory for count
        err = cudaMalloc((void**)&d_counts[i], sizeof(unsigned long long int));
        checkCudaError(err, "Allocating device memory for count");

        // Allocate device memory for next_threshold
        err = cudaMalloc((void**)&d_next_thresholds[i], sizeof(unsigned long long int));
        checkCudaError(err, "Allocating device memory for next_threshold");

        // Initialize count to zero
        err = cudaMemsetAsync(d_counts[i], 0, sizeof(unsigned long long int), streams[i]);
        checkCudaError(err, "Initializing device count to zero");

        // Initialize next_threshold to PRINT_THRESHOLD
        err = cudaMemcpyAsync(d_next_thresholds[i], &h_next_thresholds[i], sizeof(unsigned long long int), cudaMemcpyHostToDevice, streams[i]);
        checkCudaError(err, "Initializing device next_threshold");
    }

    // Step 4: Determine block and grid sizes and launch kernels
    std::vector<int> blockSizes(device_count, 256);
    std::vector<int> gridSizes(device_count, 0);

    for (int i = 0; i < device_count; ++i) {
        determine_block_grid_sizes(i, per_device_perimeters[i], blockSizes[i], gridSizes[i]);
    }

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Record start event
    cudaEventRecord(start_event, 0);

    // Launch kernels on each device
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);

        // Calculate start and end perimeters
        unsigned long long int start_p = (i * (MAX_PERIMETER / device_count)) + 1;
        unsigned long long int end_p = start_p + per_device_perimeters[i];

        // Launch kernel
        count_triangles_kernel<<<gridSizes[i], blockSizes[i], 0, streams[i]>>>(
            start_p,
            end_p,
            d_counts[i],
            d_next_thresholds[i]
        );

        // Check for kernel launch errors
        err = cudaGetLastError();
        checkCudaError(err, "Launching kernel");
    }

    // Step 5: Record stop event after all kernels are launched
    cudaEventRecord(stop_event, 0);

    // Wait for all streams to complete
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Record stop event
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    // Step 6: Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    // Step 7: Copy counts back to host
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        err = cudaMemcpyAsync(&h_counts[i], d_counts[i], sizeof(unsigned long long int), cudaMemcpyDeviceToHost, streams[i]);
        checkCudaError(err, "Copying count from device to host");

        err = cudaMemcpyAsync(&h_next_thresholds[i], d_next_thresholds[i], sizeof(unsigned long long int), cudaMemcpyDeviceToHost, streams[i]);
        checkCudaError(err, "Copying next_threshold from device to host");
    }

    // Wait for all copies to finish
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Step 8: Aggregate counts
    unsigned long long int total_count = 0;
    for (int i = 0; i < device_count; ++i) {
        total_count += h_counts[i];
    }

    // Step 9: Clean up
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaFree(d_counts[i]);
        cudaFree(d_next_thresholds[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Step 10: Output the results
    std::cout << "Total primitive triangles found: " << total_count << "\n";
    std::cout << "Execution time: " << milliseconds / 1000.0 << " seconds.\n";

    return 0;
}