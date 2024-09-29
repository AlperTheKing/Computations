#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_RESULTS 150000000 // Maximum number of triangles to store

struct Triangle {
    long long a, b, c;
};

// Efficient integer square root function for GPU
__device__ unsigned long long isqrt(unsigned long long x) {
    unsigned long long res = 0;
    unsigned long long bit = 1ULL << 62;

    while (bit > x) bit >>= 2;

    while (bit != 0) {
        if (x >= res + bit) {
            x -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

// Kernel for finding triangles
__global__ void findTrianglesKernel(int k, long long MAX_PERIMETER, Triangle* d_results, unsigned long long* d_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalThreads = gridDim.x * blockDim.x;

    long long b_min = 1;
    long long b_max = MAX_PERIMETER / 2;

    for (long long b = b_min + idx; b <= b_max; b += totalThreads) {
        for (long long c = b; c <= MAX_PERIMETER - b - 1; ++c) {
            unsigned long long D = (unsigned long long)(b - c) * (b - c) + 4 * k * b * c;
            unsigned long long s = isqrt(D);
            if (s * s != D) continue;

            long long a_numerator = - (b + c) + s;
            if (a_numerator <= 0 || a_numerator % 2 != 0) continue;
            long long a = a_numerator / 2;
            if (a > b) continue;

            if (a + b <= c || a + c <= b || b + c <= a) continue;
            if (a + b + c > MAX_PERIMETER) continue;

            unsigned long long ratio_numerator = (a + b) * (a + c);
            unsigned long long ratio_denominator = b * c;
            if (ratio_numerator != (unsigned long long)k * ratio_denominator) continue;

            unsigned long long pos = atomicAdd(d_count, 1ULL);
            if (pos < MAX_RESULTS) {
                d_results[pos].a = a;
                d_results[pos].b = b;
                d_results[pos].c = c;
            }
        }
    }
}

// Equilateral triangles kernel
__global__ void findEquilateralTrianglesKernel(long long MAX_PERIMETER, Triangle* d_results, unsigned long long* d_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalThreads = gridDim.x * blockDim.x;

    long long max_a = MAX_PERIMETER / 3;

    for (long long a = idx + 1; a <= max_a; a += totalThreads) {
        unsigned long long pos = atomicAdd(d_count, 1ULL);
        if (pos < MAX_RESULTS) {
            d_results[pos].a = a;
            d_results[pos].b = a;
            d_results[pos].c = a;
        }
    }
}

int main() {
    // User input for maximum perimeter
    long long MAX_PERIMETER;
    std::cout << "Enter the maximum perimeter: ";
    std::cin >> MAX_PERIMETER;

    auto start_time = std::chrono::high_resolution_clock::now();

    // CUDA settings
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "Error setting CUDA device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Compute optimal block size and grid size for findTrianglesKernel
    int blockSizeTriangles;
    int minGridSizeTriangles;
    size_t dynamicSMemSize = 0;

    err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSizeTriangles,
        &blockSizeTriangles,
        findTrianglesKernel,
        dynamicSMemSize,
        0);
    if (err != cudaSuccess) {
        std::cerr << "Error in cudaOccupancyMaxPotentialBlockSize for findTrianglesKernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Calculate total combinations
    long long b_min = 1;
    long long b_max = MAX_PERIMETER / 2;
    unsigned long long total_b = b_max - b_min + 1;

    // For the loops in the kernel
    unsigned long long totalThreadsNeeded = total_b;
    int gridSizeTriangles = (totalThreadsNeeded + blockSizeTriangles - 1) / blockSizeTriangles;

    // Limit grid size to maximum allowed
    int deviceMaxGridSizeX;
    err = cudaDeviceGetAttribute(&deviceMaxGridSizeX, cudaDevAttrMaxGridDimX, device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device attribute: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (gridSizeTriangles > deviceMaxGridSizeX) {
        gridSizeTriangles = deviceMaxGridSizeX;
    }

    // Print the computed block size and grid size for findTrianglesKernel
    std::cout << "Optimal block size for findTrianglesKernel: " << blockSizeTriangles << std::endl;
    std::cout << "Calculated grid size for findTrianglesKernel: " << gridSizeTriangles << std::endl;

    // Similarly, compute for findEquilateralTrianglesKernel
    int blockSizeEquilateral;
    int minGridSizeEquilateral;

    err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSizeEquilateral,
        &blockSizeEquilateral,
        findEquilateralTrianglesKernel,
        dynamicSMemSize,
        0);
    if (err != cudaSuccess) {
        std::cerr << "Error in cudaOccupancyMaxPotentialBlockSize for findEquilateralTrianglesKernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    long long max_a = MAX_PERIMETER / 3;
    unsigned long long totalThreadsNeededEquilateral = max_a;
    int gridSizeEquilateral = (totalThreadsNeededEquilateral + blockSizeEquilateral - 1) / blockSizeEquilateral;

    if (gridSizeEquilateral > deviceMaxGridSizeX) {
        gridSizeEquilateral = deviceMaxGridSizeX;
    }

    // Print the computed block size and grid size for findEquilateralTrianglesKernel
    std::cout << "Optimal block size for findEquilateralTrianglesKernel: " << blockSizeEquilateral << std::endl;
    std::cout << "Calculated grid size for findEquilateralTrianglesKernel: " << gridSizeEquilateral << std::endl;

    // Allocate memory for results
    Triangle* d_results;
    err = cudaMalloc((void**)&d_results, MAX_RESULTS * sizeof(Triangle));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for results: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    unsigned long long* d_count;
    err = cudaMalloc((void**)&d_count, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for count: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        return -1;
    }

    err = cudaMemset(d_count, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        std::cerr << "Error initializing device memory for count: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    // Launch kernels with optimized parameters
    findTrianglesKernel<<<gridSizeTriangles, blockSizeTriangles>>>(2, MAX_PERIMETER, d_results, d_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error for k=2: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    findTrianglesKernel<<<gridSizeTriangles, blockSizeTriangles>>>(3, MAX_PERIMETER, d_results, d_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error for k=3: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    findEquilateralTrianglesKernel<<<gridSizeEquilateral, blockSizeEquilateral>>>(MAX_PERIMETER, d_results, d_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error for equilateral triangles: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error during device synchronization: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    // Copy result count back to host
    unsigned long long h_count = 0;
    err = cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying count from device to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    if (h_count > MAX_RESULTS) h_count = MAX_RESULTS;

    // Copy results back to host
    std::vector<Triangle> validTriangles(h_count);
    err = cudaMemcpy(validTriangles.data(), d_results, h_count * sizeof(Triangle), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying results from device to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        cudaFree(d_count);
        return -1;
    }

    // Free device memory
    cudaFree(d_results);
    cudaFree(d_count);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    std::cout << "Found " << h_count << " valid triangles." << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl;

    // Optionally, print triangles
    /*
    for (const auto& triangle : validTriangles) {
        std::cout << "a = " << triangle.a << ", b = " << triangle.b << ", c = " << triangle.c << std::endl;
    }
    */

    return 0;
}