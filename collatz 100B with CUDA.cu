#include <iostream>
#include <vector>
#include <thread>  // Use std::thread instead of std::jthread
#include <chrono>
#include <cuda_runtime.h>
#include <mutex>

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
    if (num >= end) return;

    int steps = collatz_steps_device(num);

    atomicMax(d_max_steps, steps);
    if (*d_max_steps == steps) {
        *d_max_number = num;
    }
}

// Function executed by each thread
void process_range(long long start, long long end, long long& max_number, int& max_steps, int threads_per_block) {
    long long range_size = end - start;
    int blocks_per_grid = (range_size + threads_per_block - 1) / threads_per_block;

    // Allocate memory on the device
    int* d_max_steps;
    long long* d_max_number;
    cudaMalloc(&d_max_steps, sizeof(int));
    cudaMalloc(&d_max_number, sizeof(long long));

    // Initialize device memory
    cudaMemset(d_max_steps, 0, sizeof(int));
    cudaMemset(d_max_number, 0, sizeof(long long));

    // Launch the CUDA kernel
    collatz_kernel<<<blocks_per_grid, threads_per_block>>>(start, end, d_max_steps, d_max_number);

    // Copy results back to host
    cudaMemcpy(&max_steps, d_max_steps, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_number, d_max_number, sizeof(long long), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_max_steps);
    cudaFree(d_max_number);
}

int main() {
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

    const int threads_per_block = 256;
    const int num_threads = std::thread::hardware_concurrency();
    
    std::vector<std::thread> threads;
    std::mutex mtx;

    for (const auto& group : groups) {
        auto start_time = std::chrono::high_resolution_clock::now();

        long long max_number = 0;
        int max_steps = 0;

        auto thread_func = [&]() {
            long long local_max_number = 0;
            int local_max_steps = 0;

            process_range(group.first, group.second, local_max_number, local_max_steps, threads_per_block);

            // Update the global maximum safely
            std::lock_guard<std::mutex> lock(mtx);
            if (local_max_steps > max_steps) {
                max_steps = local_max_steps;
                max_number = local_max_number;
            }
        };

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(thread_func);
        }

        // Join all threads
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

    return 0;
}