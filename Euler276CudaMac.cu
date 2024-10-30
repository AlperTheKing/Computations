#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <atomic>
#include <stdio.h>      // Required for fprintf and stderr
#include <vector>       // Required for std::vector
#include <tuple>        // Required for std::tuple

#define MAX_PERIMETER 10000000
#define TRIANGLE_STORAGE_INTERVAL 1000000
#define MAX_TRIANGLES_TO_STORE 10 // Adjust based on expected count

// Macro to check CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Function to handle CUDA errors
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Device-side variables (separate for each GPU)
__device__ unsigned long long triangle_count;

// Arrays to store every millionth triangle
__device__ unsigned long long triangle_data_a[MAX_TRIANGLES_TO_STORE];
__device__ unsigned long long triangle_data_b[MAX_TRIANGLES_TO_STORE];
__device__ unsigned long long triangle_data_c[MAX_TRIANGLES_TO_STORE];

// Device GCD function
__device__ int64_t gcd(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

// Kernel to count primitive triangles
__global__ void count_triangles(int64_t start, int64_t end) {
    int64_t p = blockIdx.x * blockDim.x + threadIdx.x + start;

    if (p < end) {
        for (int64_t a = 1; a <= p / 3; ++a) {
            for (int64_t b = a; b <= (p - a) / 2; ++b) {
                int64_t c = p - a - b;
                if (c >= b && a + b > c) { // Triangle inequality
                    if (gcd(gcd(a, b), c) == 1) { // Primitive triangle
                        unsigned long long count = atomicAdd(&triangle_count, 1);
                        if ((count + 1) % TRIANGLE_STORAGE_INTERVAL == 0 && 
                            (count + 1) / TRIANGLE_STORAGE_INTERVAL <= MAX_TRIANGLES_TO_STORE) {
                            int index = (count + 1) / TRIANGLE_STORAGE_INTERVAL - 1;
                            triangle_data_a[index] = a;
                            triangle_data_b[index] = b;
                            triangle_data_c[index] = c;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Get CUDA device count
    int deviceCount;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        std::cerr << "No CUDA devices found.\n";
        return -1;
    }
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    // Display device names
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaCheckError(cudaGetDeviceProperties(&deviceProp, i));
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Structure to hold per-device data
    struct DeviceData {
        int deviceId;
        cudaStream_t stream;
        unsigned long long h_triangle_count;
        unsigned long long h_triangle_data_a[MAX_TRIANGLES_TO_STORE];
        unsigned long long h_triangle_data_b[MAX_TRIANGLES_TO_STORE];
        unsigned long long h_triangle_data_c[MAX_TRIANGLES_TO_STORE];
        dim3 gridSize;
        dim3 blockSize;
    };

    std::vector<DeviceData> devices(deviceCount);

    // Determine per-device workload and initialize each device
    for (int i = 0; i < deviceCount; ++i) {
        devices[i].deviceId = i;
        cudaCheckError(cudaSetDevice(i));

        // Initialize triangle_count to zero on device
        unsigned long long zero = 0;
        cudaCheckError(cudaMemcpyToSymbol(triangle_count, &zero, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));

        // Determine optimal block and grid sizes for this device
        int minGridSize;
        int blockSizeMax;
        cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax, count_triangles, 0, 0));
        devices[i].blockSize = dim3(blockSizeMax);
        
        // Calculate grid size based on per-device workload
        int64_t perDevice = MAX_PERIMETER / deviceCount;
        int64_t start_p = perDevice * i + 1;
        int64_t end_p = (i == deviceCount - 1) ? MAX_PERIMETER + 1 : perDevice * (i + 1) + 1;
        int64_t totalWork = end_p - start_p;
        devices[i].gridSize = dim3((totalWork + devices[i].blockSize.x - 1) / devices[i].blockSize.x);

        // Create a CUDA stream for this device
        cudaCheckError(cudaStreamCreate(&devices[i].stream));
    }

    // Launch kernels on each device
    for (int i = 0; i < deviceCount; ++i) {
        cudaCheckError(cudaSetDevice(devices[i].deviceId));

        int64_t perDevice = MAX_PERIMETER / deviceCount;
        int64_t start_p = perDevice * i + 1;
        int64_t end_p = (i == deviceCount - 1) ? MAX_PERIMETER + 1 : perDevice * (i + 1) + 1;

        // Launch the kernel on this device's stream
        count_triangles<<<devices[i].gridSize, devices[i].blockSize, 0, devices[i].stream>>>(start_p, end_p);
        cudaCheckError(cudaGetLastError());
    }

    // Synchronize all devices
    for (int i = 0; i < deviceCount; ++i) {
        cudaCheckError(cudaSetDevice(devices[i].deviceId));
        cudaCheckError(cudaStreamSynchronize(devices[i].stream));
    }

    // Collect results from all devices
    unsigned long long total_triangle_count = 0;
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> all_triangles;

    for (int i = 0; i < deviceCount; ++i) {
        cudaCheckError(cudaSetDevice(devices[i].deviceId));

        // Copy triangle_count from device to host
        cudaCheckError(cudaMemcpyFromSymbol(&devices[i].h_triangle_count, triangle_count, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
        total_triangle_count += devices[i].h_triangle_count;

        // Copy triangle_data from device to host
        cudaCheckError(cudaMemcpy(devices[i].h_triangle_data_a, triangle_data_a, sizeof(devices[i].h_triangle_data_a), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(devices[i].h_triangle_data_b, triangle_data_b, sizeof(devices[i].h_triangle_data_b), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(devices[i].h_triangle_data_c, triangle_data_c, sizeof(devices[i].h_triangle_data_c), cudaMemcpyDeviceToHost));

        // Aggregate stored triangles
        unsigned long long num_million = devices[i].h_triangle_count / TRIANGLE_STORAGE_INTERVAL;
        for (unsigned long long j = 0; j < num_million && j < MAX_TRIANGLES_TO_STORE; ++j) {
            all_triangles.emplace_back(
                devices[i].h_triangle_data_a[j],
                devices[i].h_triangle_data_b[j],
                devices[i].h_triangle_data_c[j]
            );
        }
    }

    // Output every millionth triangle
    for (size_t i = 0; i < all_triangles.size(); ++i) {
        auto [a, b, c] = all_triangles[i];
        std::cout << "Found triangle #" << (i + 1) * TRIANGLE_STORAGE_INTERVAL << ": ("
                  << a << ", " << b << ", " << c << ")\n";
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Total primitive triangles found: " << total_triangle_count << "\n";
    std::cout << "Execution time: " << duration.count() << " seconds.\n";

    // Clean up streams
    for (int i = 0; i < deviceCount; ++i) {
        cudaCheckError(cudaSetDevice(devices[i].deviceId));
        cudaCheckError(cudaStreamDestroy(devices[i].stream));
    }

    return 0;
}