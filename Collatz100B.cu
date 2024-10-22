#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <thread>
#include <iomanip>

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string BOLD_RED = "\033[1;31m";
const std::string BOLD_GREEN = "\033[1;32m";
const std::string BOLD_YELLOW = "\033[1;33m";
const std::string BOLD_BLUE = "\033[1;34m";

struct uint128 {
    unsigned long long low;
    unsigned long long high;
};

__host__ __device__ uint128 add_uint128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low);
    return result;
}

__host__ __device__ uint128 mul_uint128(unsigned long long a, unsigned long long b) {
    uint128 result;
    unsigned __int128 full_result = static_cast<unsigned __int128>(a) * b;
    result.low = static_cast<unsigned long long>(full_result);
    result.high = static_cast<unsigned long long>(full_result >> 64);
    return result;
}

__device__ void atomicMax_uint128(unsigned long long *d_max_steps, uint128 *d_number_with_max_steps, unsigned long long local_max_steps, uint128 local_number_with_max_steps) {
    unsigned long long prev_max_steps = atomicMax(d_max_steps, local_max_steps);
    __syncthreads();  // Ensure all threads see the updated max steps value
    if (local_max_steps > prev_max_steps) {
        atomicExch(&(d_number_with_max_steps->low), local_number_with_max_steps.low);
        atomicExch(&(d_number_with_max_steps->high), local_number_with_max_steps.high);
    }
}

__global__ void find_max_collatz_steps_in_range(uint128 start, uint128 end, unsigned long long *d_max_steps, uint128 *d_number_with_max_steps) {
    unsigned long long local_max_steps = 0;
    uint128 local_number_with_max_steps = start;

    uint128 current = start;
    current.low += blockIdx.x * blockDim.x + threadIdx.x;

    while (current.low < end.low || current.high < end.high) {
        unsigned long long steps = 0;
        unsigned long long low = current.low;
        unsigned long long high = current.high;
        while (low != 1 || high != 0) {
            if (high == 0) {  // Use 64-bit logic if within 64-bit range
                if (low % 2 == 0) {
                    low /= 2;
                } else {
                    low = 3 * low + 1;
                }
            } else {  // Use 128-bit logic if exceeding 64-bit range
                if ((low & 1) == 0) {  // Even number
                    low >>= 1;
                    if (high & 1) {
                        low |= (1ULL << 63);
                    }
                    high >>= 1;
                } else {  // Odd number
                    uint128 three_n = mul_uint128(low, 3);
                    three_n.high += high * 3;
                    current = add_uint128(three_n, {1, 0});
                    low = current.low;
                    high = current.high;
                }
            }
            steps++;
        }
        if (steps > local_max_steps) {
            local_max_steps = steps;
            local_number_with_max_steps = current;
        }
        current.low += gridDim.x * blockDim.x;
        if (current.low < blockIdx.x * blockDim.x + threadIdx.x) {
            current.high++;
        }
    }

    atomicMax_uint128(d_max_steps, d_number_with_max_steps, local_max_steps, local_number_with_max_steps);
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    // Define the ranges for the full 128-bit limit
    std::vector<std::pair<uint128, uint128>> ranges;
    uint128 base = {1, 0};  // Start from 2^0 to cover the full 128-bit range
    for (int i = 0; i < 40; ++i) {
        uint128 start = base;
        uint128 end = base;
        uint128 multiplier = {10, 0};
        uint128 result = mul_uint128(base.low, multiplier.low);
        end.low = result.low;
        end.high = base.high * multiplier.low + result.high;
        ranges.emplace_back(start, end);
        base = end;
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop through each range and find the number with the maximum Collatz steps
    for (int i = 0; i < ranges.size(); ++i) {
        const auto &range = ranges[i];
        uint128 start = range.first;
        uint128 end = range.second;

        std::vector<cudaStream_t> streams(device_count);
        std::vector<unsigned long long *> d_max_steps(device_count);
        std::vector<uint128 *> d_number_with_max_steps(device_count);

        unsigned long long h_max_steps = 0;
        uint128 h_number_with_max_steps = {0, 0};

        for (int device = 0; device < device_count; ++device) {
            cudaSetDevice(device);

            cudaMalloc(&d_max_steps[device], sizeof(unsigned long long));
            cudaMalloc(&d_number_with_max_steps[device], sizeof(uint128));

            cudaMemset(d_max_steps[device], 0, sizeof(unsigned long long));
            cudaMemset(d_number_with_max_steps[device], 0, sizeof(uint128));

            cudaStreamCreate(&streams[device]);

            int threads_per_block;
            int blocks_per_grid;
            cudaOccupancyMaxPotentialBlockSize(&blocks_per_grid, &threads_per_block, find_max_collatz_steps_in_range);

            find_max_collatz_steps_in_range<<<blocks_per_grid, threads_per_block, 0, streams[device]>>>(start, end, d_max_steps[device], d_number_with_max_steps[device]);
        }

        for (int device = 0; device < device_count; ++device) {
            cudaSetDevice(device);
            cudaStreamSynchronize(streams[device]);

            unsigned long long temp_max_steps;
            uint128 temp_number_with_max_steps;
            cudaMemcpy(&temp_max_steps, d_max_steps[device], sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&temp_number_with_max_steps, d_number_with_max_steps[device], sizeof(uint128), cudaMemcpyDeviceToHost);

            if (temp_max_steps > h_max_steps) {
                h_max_steps = temp_max_steps;
                h_number_with_max_steps = temp_number_with_max_steps;
            }

            cudaFree(d_max_steps[device]);
            cudaFree(d_number_with_max_steps[device]);
            cudaStreamDestroy(streams[device]);
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> execution_time = end_time - start_time;

        // Display the result vertically with color
        std::cout << BOLD_BLUE << "Range " << start.low << " - " << end.low << ":\n"
                  << BOLD_GREEN << "Number with max steps: " << h_number_with_max_steps.low << " (high: " << h_number_with_max_steps.high << ")\n"
                  << BOLD_YELLOW << "Steps: " << h_max_steps << "\n"
                  << BOLD_RED << "Time taken: " << execution_time.count() << " seconds\n"
                  << RESET << "-----------------------------\n";
    }

    return 0;
}