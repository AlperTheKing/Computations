// Filename: fourth_power_equation_multi_gpu.cu

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint> // Include standard integer types

#define MAX_N 100000  // Set MAX_N to 100,000 as per your request

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Structure to simulate 128-bit integers using two 64-bit integers
struct uint128_t {
    uint64_t low;
    uint64_t high;

    __host__ __device__
    uint128_t() : low(0), high(0) {}

    __host__ __device__
    uint128_t(uint64_t l, uint64_t h) : low(l), high(h) {}

    // Addition
    __host__ __device__
    uint128_t operator+(const uint128_t& other) const {
        uint64_t new_low = low + other.low;
        uint64_t carry = (new_low < low) ? 1 : 0;
        uint64_t new_high = high + other.high + carry;
        return uint128_t(new_low, new_high);
    }

    // Subtraction
    __host__ __device__
    uint128_t operator-(const uint128_t& other) const {
        uint64_t new_low = low - other.low;
        uint64_t borrow = (low < other.low) ? 1 : 0;
        uint64_t new_high = high - other.high - borrow;
        return uint128_t(new_low, new_high);
    }

    // Comparison operators
    __host__ __device__
    bool operator<(const uint128_t& other) const {
        return (high < other.high) || (high == other.high && low < other.low);
    }

    __host__ __device__
    bool operator>(const uint128_t& other) const {
        return (high > other.high) || (high == other.high && low > other.low);
    }

    __host__ __device__
    bool operator==(const uint128_t& other) const {
        return high == other.high && low == other.low;
    }
};

// Host function to multiply two 64-bit integers and get a 128-bit result
void multiply_64x64(uint64_t a, uint64_t b, uint64_t& high, uint64_t& low) {
    __uint128_t result = (__uint128_t)a * (__uint128_t)b;
    low = (uint64_t)(result & 0xFFFFFFFFFFFFFFFFULL);
    high = (uint64_t)(result >> 64);
}

// Kernel function
__global__ void find_solutions(uint64_t* d_fourth_powers_low, uint64_t* d_fourth_powers_high, int max_n, int start_e, int end_e) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int e = start_e + idx;

    if (e > end_e || e > max_n || e <= 0) return; // Ensure e is within valid range

    uint128_t e4 = uint128_t(d_fourth_powers_low[e - 1], d_fourth_powers_high[e - 1]);

    for (int d = 1; d <= e && d <= max_n; ++d) {
        uint128_t d4 = uint128_t(d_fourth_powers_low[d - 1], d_fourth_powers_high[d - 1]);
        if (d4 > e4) break;

        for (int c = 1; c <= d && c <= max_n; ++c) {
            uint128_t c4 = uint128_t(d_fourth_powers_low[c - 1], d_fourth_powers_high[c - 1]);
            if (d4 + c4 > e4) break;

            uint128_t s = e4 - d4 - c4;

            // Since a ≤ b ≤ c, and a, b ≤ c
            for (int a = 1; a <= c && a <= max_n; ++a) {
                uint128_t a4 = uint128_t(d_fourth_powers_low[a - 1], d_fourth_powers_high[a - 1]);
                if (a4 > s) break;
                uint128_t b4 = s - a4;

                // Binary search for b in [a, c] such that b^4 == b4
                int left = a;
                int right = c;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (mid > max_n) {
                        right = mid - 1;
                        continue;
                    }

                    uint128_t b_candidate4 = uint128_t(d_fourth_powers_low[mid - 1], d_fourth_powers_high[mid - 1]);

                    if (b_candidate4 == b4) {
                        // Found a solution
                        printf("Solution found: (a, b, c, d, e) = (%d, %d, %d, %d, %d)\n", a, mid, c, d, e);
                        break;
                    } else if (b_candidate4 < b4) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            }
        }
    }
}

int main() {
    // Precompute fourth powers on host
    std::vector<uint128_t> fourth_powers(MAX_N);
    for (int i = 0; i < MAX_N; ++i) {
        uint64_t i64 = static_cast<uint64_t>(i + 1); // i from 1 to MAX_N
        uint64_t i2_low, i2_high;
        multiply_64x64(i64, i64, i2_high, i2_low); // i^2

        // Compute i^4 = (i^2)^2
        uint64_t i4_low, i4_high;
        multiply_64x64(i2_low, i2_low, i4_high, i4_low);

        // Add the cross terms
        uint64_t cross_low, cross_high;
        multiply_64x64(i2_low, i2_high * 2, cross_high, cross_low);

        uint64_t final_low = i4_low + (cross_low << 1);
        uint64_t carry = (final_low < i4_low) ? 1 : 0;
        uint64_t final_high = i4_high + (cross_high << 1) + carry;

        fourth_powers[i] = uint128_t(final_low, final_high);
    }

    // Get the number of devices
    int num_devices;
    cudaCheckError(cudaGetDeviceCount(&num_devices));
    std::cout << "Number of CUDA devices detected: " << num_devices << std::endl;
    if (num_devices < 2) {
        std::cerr << "Warning: Less than 2 CUDA devices detected. Adjusting to use available devices." << std::endl;
    }
    num_devices = std::min(num_devices, 2); // Use up to 2 GPUs

    // Measure execution time
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // Start timer
    cudaCheckError(cudaEventRecord(start, 0));

    // Assign e ranges as per your request
    int start_e_array[] = {1, 67001};
    int end_e_array[] = {67000, MAX_N};

    std::vector<cudaStream_t> streams(num_devices);
    std::vector<uint64_t*> d_fourth_powers_low(num_devices);
    std::vector<uint64_t*> d_fourth_powers_high(num_devices);

    // Use per-device memory allocations and copies
    for (int device = 0; device < num_devices; ++device) {
        cudaCheckError(cudaSetDevice(device));

        // Allocate device memory
        size_t size = MAX_N * sizeof(uint64_t);
        cudaCheckError(cudaMalloc((void**)&d_fourth_powers_low[device], size));
        cudaCheckError(cudaMalloc((void**)&d_fourth_powers_high[device], size));

        // Copy data to device
        cudaCheckError(cudaMemcpy(d_fourth_powers_low[device], &fourth_powers[0].low, size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fourth_powers_high[device], &fourth_powers[0].high, size, cudaMemcpyHostToDevice));

        // Create stream
        cudaCheckError(cudaStreamCreate(&streams[device]));
    }

    // Set block size to 256 as per your request
    int blockSize = 256;

    for (int device = 0; device < num_devices; ++device) {
        cudaCheckError(cudaSetDevice(device));

        int current_device;
        cudaCheckError(cudaGetDevice(&current_device));
        printf("Launching kernel on Device %d\n", current_device);

        int start_e = start_e_array[device];
        int end_e = end_e_array[device];

        if (end_e > MAX_N) end_e = MAX_N;

        // Adjust grid size for each device
        int num_elements = end_e - start_e + 1;
        int gridSize = (num_elements + blockSize - 1) / blockSize;

        if (gridSize < 1) gridSize = 1;

        printf("Device %d: start_e = %d, end_e = %d, gridSize = %d\n", device, start_e, end_e, gridSize);

        // Launch kernel on each device
        find_solutions<<<gridSize, blockSize, 0, streams[device]>>>(d_fourth_powers_low[device], d_fourth_powers_high[device], MAX_N, start_e, end_e);

        // Check for kernel launch errors
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            fprintf(stderr, "Kernel launch error on device %d: %s\n", device, cudaGetErrorString(kernelErr));
            exit(EXIT_FAILURE);
        }
    }

    // Synchronize streams and devices
    for (int device = 0; device < num_devices; ++device) {
        cudaCheckError(cudaSetDevice(device));
        cudaCheckError(cudaStreamSynchronize(streams[device]));
    }

    // Stop timer
    cudaCheckError(cudaEventRecord(stop, 0));
    cudaCheckError(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    int total_seconds = static_cast<int>(milliseconds / 1000.0f);
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;

    std::cout << "Elapsed time: " << hours << "h " << minutes << "m " << seconds << "s\n";

    // Clean up
    for (int device = 0; device < num_devices; ++device) {
        cudaCheckError(cudaSetDevice(device));
        cudaCheckError(cudaStreamDestroy(streams[device]));
        cudaCheckError(cudaFree(d_fourth_powers_low[device]));
        cudaCheckError(cudaFree(d_fourth_powers_high[device]));
    }

    return 0;
}