// CollatzMultiGPU_Ranged.cu

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <thread>
#include <mutex>
#include <sstream>
#include <algorithm>

// ANSI color codes for colorful output (optional)
const std::string RESET = "\033[0m";
const std::string BOLD_RED = "\033[1;31m";
const std::string BOLD_GREEN = "\033[1;32m";
const std::string BOLD_YELLOW = "\033[1;33m";
const std::string BOLD_BLUE = "\033[1;34m";

// Structure to represent 128-bit unsigned integers
struct uint128 {
    unsigned long long low;
    unsigned long long high;
};

// Host and device functions for 128-bit arithmetic

__host__ __device__ uint128 add_uint128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low ? 1ULL : 0ULL);
    return result;
}

__host__ __device__ uint128 subtract_uint128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low - b.low;
    result.high = a.high - b.high - (a.low < b.low ? 1ULL : 0ULL);
    return result;
}

__host__ __device__ uint128 right_shift_uint128(uint128 a) {
    uint128 result;
    result.low = (a.low >> 1) | (a.high << 63);
    result.high = a.high >> 1;
    return result;
}

__host__ __device__ bool is_even_uint128(uint128 a) {
    return (a.low & 1ULL) == 0ULL;
}

__host__ __device__ bool less_than_uint128(uint128 a, uint128 b) {
    return (a.high < b.high) || (a.high == b.high && a.low < b.low);
}

__host__ __device__ bool less_than_or_equal_uint128(uint128 a, uint128 b) {
    return (a.high < b.high) || (a.high == b.high && a.low <= b.low);
}

__host__ __device__ void increment_uint128(uint128* value, unsigned long long increment) {
    unsigned long long old_low = value->low;
    value->low += increment;
    if (value->low < old_low) {
        value->high += 1;
    }
}

// Modified multiply_uint128 function
__host__ __device__ uint128 multiply_uint128(uint128 a, unsigned long long b) {
    uint128 result = {0ULL, 0ULL};

    unsigned long long a_low = a.low;
    unsigned long long a_high = a.high;

    unsigned int al_low = (unsigned int)(a_low & 0xFFFFFFFFULL);
    unsigned int al_high = (unsigned int)(a_low >> 32);

    unsigned int b_low = (unsigned int)(b & 0xFFFFFFFFULL);
    unsigned int b_high = (unsigned int)(b >> 32);

    // Compute partial products
    unsigned long long ll = (unsigned long long)al_low * b_low;
    unsigned long long lh = (unsigned long long)al_low * b_high;
    unsigned long long hl = (unsigned long long)al_high * b_low;
    unsigned long long hh = (unsigned long long)al_high * b_high;

    // Combine partial products
    unsigned long long mid = lh + hl;
    unsigned long long carry = (mid < lh) ? (1ULL << 32) : 0ULL;

    result.low = ll + (mid << 32);
    if (result.low < ll) carry++;

    result.high = a_high * b + hh + (mid >> 32) + carry;

    return result;
}

// Function to divide uint128 by an unsigned int
__host__ uint128 divide_uint128(uint128 dividend, unsigned int divisor) {
    uint128 result = {0ULL, 0ULL};
    uint128 remainder = {0ULL, 0ULL};

    for (int i = 127; i >= 0; --i) {
        // Left shift remainder by 1
        remainder.high = (remainder.high << 1) | (remainder.low >> 63);
        remainder.low = (remainder.low << 1);

        // Bring down the next bit of the dividend
        if (i >= 64) {
            remainder.low |= (dividend.high >> (i - 64)) & 1ULL;
        } else {
            remainder.low |= (dividend.low >> i) & 1ULL;
        }

        // If remainder >= divisor
        if (remainder.high > 0 || remainder.low >= divisor) {
            remainder.low -= divisor;
            // Set the corresponding bit in the result
            if (i >= 64) {
                result.high |= (1ULL << (i - 64));
            } else {
                result.low |= (1ULL << i);
            }
        }
    }
    return result;
}

// Function to compute uint128 power of 10
__host__ uint128 uint128_pow10(unsigned int exponent) {
    uint128 result = {1ULL, 0ULL};

    for (unsigned int i = 0; i < exponent; ++i) {
        result = multiply_uint128(result, 10ULL);
    }
    return result;
}

// Simplified print_uint128 function
void print_uint128(uint128 a) {
    if (a.high == 0ULL) {
        std::cout << a.low;
    } else {
        // For large numbers, print high and low parts
        std::cout << a.high << std::setw(20) << std::setfill('0') << a.low;
    }
}

// CUDA error-checking macro
#define cudaCheckError(call)                                    \
    {                                                           \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            std::cerr << "CUDA error in " << __FILE__           \
                      << " at line " << __LINE__ << ": "        \
                      << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    }

// Atomic function to update the maximum steps
__device__ void update_max_steps(unsigned long long *max_steps, uint128 *number_with_max_steps,
                                 unsigned long long local_steps, uint128 local_number) {
    unsigned long long prev_max_steps = atomicMax(max_steps, local_steps);

    if (local_steps > prev_max_steps) {
        unsigned long long* addr = (unsigned long long*)number_with_max_steps;
        atomicExch(addr, local_number.low);
        atomicExch(addr + 1, local_number.high);
    }
}

// Kernel function to find the number with the maximum Collatz steps in a given range
__global__ void find_max_collatz_steps_in_range(uint128 start, uint128 end,
                                                unsigned long long *d_max_steps,
                                                uint128 *d_number_with_max_steps) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    uint128 current = start;
    increment_uint128(&current, idx);

    unsigned long long local_max_steps = 0;
    uint128 local_number_with_max_steps = current;

    while (less_than_or_equal_uint128(current, end)) {
        uint128 n = current;
        unsigned long long steps = 0;

        // Compute Collatz steps
        while (!(n.low == 1ULL && n.high == 0ULL)) {
            if (is_even_uint128(n)) {
                n = right_shift_uint128(n);
            } else {
                n = add_uint128(multiply_uint128(n, 3ULL), {1ULL, 0ULL});
            }
            steps++;
        }

        if (steps > local_max_steps) {
            local_max_steps = steps;
            local_number_with_max_steps = current;
        }

        increment_uint128(&current, stride);
    }

    // Update global maximum
    update_max_steps(d_max_steps, d_number_with_max_steps, local_max_steps, local_number_with_max_steps);
}

// Host function to perform computation on each GPU
void gpu_compute(int gpu_id, uint128 start, uint128 end,
                 unsigned long long &h_max_steps, uint128 &h_number_with_max_steps) {
    // Set the device for this thread
    cudaCheckError(cudaSetDevice(gpu_id));

    // Prepare device memory
    unsigned long long *d_max_steps;
    uint128 *d_number_with_max_steps;

    cudaCheckError(cudaMalloc(&d_max_steps, sizeof(unsigned long long)));
    cudaCheckError(cudaMalloc(&d_number_with_max_steps, sizeof(uint128)));

    cudaCheckError(cudaMemset(d_max_steps, 0, sizeof(unsigned long long)));
    cudaCheckError(cudaMemset(d_number_with_max_steps, 0, sizeof(uint128)));

    // Determine optimal block size and grid size
    int device;
    cudaDeviceProp deviceProp;
    cudaCheckError(cudaGetDevice(&device));
    cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

    int blockSize = 256; // Adjust based on your GPU
    int gridSize = (deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount) / blockSize;

    // Debug: Print GPU ID and assigned range
    std::cout << "GPU " << gpu_id << " processing range: Start = ";
    print_uint128(start);
    std::cout << ", End = ";
    print_uint128(end);
    std::cout << std::endl;

    // Launch kernel
    find_max_collatz_steps_in_range<<<gridSize, blockSize>>>(start, end, d_max_steps, d_number_with_max_steps);

    // Synchronize
    cudaCheckError(cudaDeviceSynchronize());

    // Copy results back to host
    cudaCheckError(cudaMemcpy(&h_max_steps, d_max_steps, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(&h_number_with_max_steps, d_number_with_max_steps, sizeof(uint128), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaCheckError(cudaFree(d_max_steps));
    cudaCheckError(cudaFree(d_number_with_max_steps));
}

int main() {
    std::cout << "Program started." << std::endl;

    // Get device count
    int device_count = 0;
    cudaCheckError(cudaGetDeviceCount(&device_count));

    std::cout << "Number of CUDA devices: " << device_count << std::endl;

    if (device_count < 1) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Generate powers of 10 from 10^0 to 10^n (n <= 38 due to uint128 limitations)
    const unsigned int max_exponent = 20; // Adjust as needed, max 38
    std::vector<uint128> powers_of_10;

    for (unsigned int i = 0; i <= max_exponent + 1; ++i) {
        uint128 power = uint128_pow10(i);
        powers_of_10.push_back(power);
    }

    // Start timing
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Loop over each range
    for (size_t range_idx = 0; range_idx < powers_of_10.size() - 1; ++range_idx) {
        uint128 start = powers_of_10[range_idx];
        uint128 end = subtract_uint128(powers_of_10[range_idx + 1], {1ULL, 0ULL});

        // Calculate the total range
        uint128 total_range = subtract_uint128(end, start);
        increment_uint128(&total_range, 1ULL); // total_range = end - start + 1

        // Split the total range among GPUs
        std::vector<std::pair<uint128, uint128>> gpu_ranges(device_count);

        // Calculate the size of each subrange
        uint128 range_size = divide_uint128(total_range, device_count);

        uint128 one = {1ULL, 0ULL};

        for (int i = 0; i < device_count; ++i) {
            gpu_ranges[i].first = add_uint128(start, multiply_uint128(range_size, i));
            if (i == device_count - 1) {
                gpu_ranges[i].second = end;
            } else {
                gpu_ranges[i].second = subtract_uint128(add_uint128(gpu_ranges[i].first, range_size), one);
            }
        }

        // Variables to hold the maximum steps and corresponding numbers from each GPU
        std::vector<unsigned long long> max_steps_per_gpu(device_count, 0);
        std::vector<uint128> number_with_max_steps_per_gpu(device_count);

        std::cout << BOLD_BLUE << "Processing range: ";
        print_uint128(start);
        std::cout << " to ";
        print_uint128(end);
        std::cout << RESET << std::endl;

        // Start timing for this range
        auto range_start_time = std::chrono::high_resolution_clock::now();

        // Create and start threads for each GPU
        std::vector<std::thread> gpu_threads(device_count);
        for (int i = 0; i < device_count; ++i) {
            gpu_threads[i] = std::thread([&, i]() {
                // Each thread sets its own device
                cudaCheckError(cudaSetDevice(i));
                gpu_compute(i, gpu_ranges[i].first, gpu_ranges[i].second,
                            std::ref(max_steps_per_gpu[i]),
                            std::ref(number_with_max_steps_per_gpu[i]));
            });
        }

        // Wait for all threads to finish
        for (int i = 0; i < device_count; ++i) {
            gpu_threads[i].join();
        }

        std::cout << "GPU computations completed for this range." << std::endl;

        // Find the overall maximum steps and corresponding number
        unsigned long long h_max_steps = 0;
        uint128 h_number_with_max_steps = {0ULL, 0ULL};
        for (int i = 0; i < device_count; ++i) {
            if (max_steps_per_gpu[i] > h_max_steps) {
                h_max_steps = max_steps_per_gpu[i];
                h_number_with_max_steps = number_with_max_steps_per_gpu[i];
            }
        }

        // End timing for this range
        auto range_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> execution_time = range_end_time - range_start_time;

        // Display the result for this range
        std::cout << BOLD_GREEN << "Number with max steps: ";
        print_uint128(h_number_with_max_steps);
        std::cout << RESET << std::endl;

        std::cout << BOLD_YELLOW << "Steps: " << h_max_steps << RESET << std::endl;
        std::cout << BOLD_RED << "Time taken for this range: " << execution_time.count() << " seconds" << RESET << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }

    // End timing
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_execution_time = total_end_time - total_start_time;

    std::cout << BOLD_RED << "Total time taken: " << total_execution_time.count() << " seconds" << RESET << std::endl;

    return 0;
}
