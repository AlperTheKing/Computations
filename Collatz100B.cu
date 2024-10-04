#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <mutex>

#define CUDA_CALL(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA Kernel for Collatz steps calculation
__device__ int collatz_steps_device(long long n) {
    int steps = 0;
    while (n != 1) {
        if (n & 1) {
            n = 3 * n + 1;
        } else {
            n = n >> 1;
        }
        steps++;
    }
    return steps;
}

// Kernel to compute Collatz on GPU and find the max steps
__global__ void collatz_kernel(long long start, long long end, int* d_max_steps, long long* d_max_number) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long num = start + index;

    if (num >= end) return;  // Ensure the thread does not process out-of-bounds values

    int steps = collatz_steps_device(num);

    // Atomic update for max steps and number
    int old_steps = atomicMax(d_max_steps, steps);
    if (steps > old_steps) {
        *d_max_number = num;
    }
}

// Function to launch GPU kernel and handle streams on multiple GPUs
void process_range_gpu(int gpu_id, long long start, long long end, long long& max_number, int& max_steps, cudaStream_t stream) {
    // Set the device for this GPU
    CUDA_CALL(cudaSetDevice(gpu_id));

    long long range_size = end - start;
    
    // Determine optimal block and grid size using occupancy calculator
    int threads_per_block = 0;
    int min_grid_size = 0;
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &threads_per_block, collatz_kernel, 0, range_size));

    int blocks_per_grid = (range_size + threads_per_block - 1) / threads_per_block;

    // Allocate memory on the device
    int* d_max_steps;
    long long* d_max_number;
    CUDA_CALL(cudaMalloc(&d_max_steps, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_max_number, sizeof(long long)));

    // Initialize device memory
    CUDA_CALL(cudaMemsetAsync(d_max_steps, 0, sizeof(int), stream));
    CUDA_CALL(cudaMemsetAsync(d_max_number, 0, sizeof(long long), stream));

    // Launch the CUDA kernel asynchronously on the stream
    collatz_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(start, end, d_max_steps, d_max_number);
    
    // Synchronize the stream and check for errors
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaGetLastError());

    // Copy results back to host
    CUDA_CALL(cudaMemcpyAsync(&max_steps, d_max_steps, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaMemcpyAsync(&max_number, d_max_number, sizeof(long long), cudaMemcpyDeviceToHost, stream));

    // Free device memory
    CUDA_CALL(cudaFree(d_max_steps));
    CUDA_CALL(cudaFree(d_max_number));
}

int main() {
    // Check the number of available GPUs
    int num_gpus;
    CUDA_CALL(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        std::cerr << "This program requires at least 2 GPUs." << std::endl;
        return 1;
    }

    std::vector<std::pair<long long, long long>> groups = {
        {1, 10},
        {10, 100},
        {100, 1000},
        {1000, 10000},
        {10000, 100000},
        {100000, 1000000},
        {1000000, 10000000},
        {10000000, 100000000},
        {100000000, 1000000000},
        {1000000000, 10000000000},
        {10000000000, 100000000000}
    };

    const int num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    std::mutex mtx;

    // Each GPU will use a stream for asynchronous kernel execution
    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    for (const auto& group : groups) {
        auto start_time = std::chrono::high_resolution_clock::now();

        long long max_number = 0;
        int max_steps = 0;

        // Divide the range between two GPUs
        long long range_mid = (group.second - group.first) / 2;
        long long gpu1_start = group.first;
        long long gpu1_end = group.first + range_mid;
        long long gpu2_start = group.first + range_mid;
        long long gpu2_end = group.second;

        // Thread function to run on each GPU
        auto thread_func = [&](int gpu_id, long long start, long long end, cudaStream_t stream) {
            long long local_max_number = 0;
            int local_max_steps = 0;

            process_range_gpu(gpu_id, start, end, local_max_number, local_max_steps, stream);

            // Update the global maximum safely
            std::lock_guard<std::mutex> lock(mtx);
            if (local_max_steps > max_steps) {
                max_steps = local_max_steps;
                max_number = local_max_number;
            }
        };

        // Launch a thread for each GPU
        threads.emplace_back(thread_func, 0, gpu1_start, gpu1_end, streams[0]);
        threads.emplace_back(thread_func, 1, gpu2_start, gpu2_end, streams[1]);

        // Join both threads
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Range " << group.first << " to " << group.second << ":\n";
        std::cout << "  Number with max steps: " << max_number << " (" << max_steps << " steps)\n";
        std::cout << "Computation time: " << elapsed.count() << " seconds\n\n";
    }

    // Destroy CUDA streams
    for (int i = 0; i < 2; ++i) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    }

    return 0;
}