#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_RESULTS 100000000 // Maximum number of triangles to store

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

    // Calculate total combinations
    unsigned long long total_b = b_max - b_min + 1;
    unsigned long long total_combinations = total_b * (total_b + 1) / 2;

    for (unsigned long long comb_idx = idx; comb_idx < total_combinations; comb_idx += totalThreads) {
        // Map comb_idx to (b, c)
        long long b = b_min + (long long)((-1 + sqrt(1 + 8 * comb_idx)) / 2);
        long long temp = comb_idx - (b - b_min) * (b - b_min + 1) / 2;
        long long c = b + temp;

        if (c > MAX_PERIMETER - b - 1) continue;

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
    cudaSetDevice(device);

    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;

    // Allocate memory for results
    Triangle* d_results;
    cudaMalloc((void**)&d_results, MAX_RESULTS * sizeof(Triangle));

    unsigned long long* d_count;
    cudaMalloc((void**)&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    // Launch kernels for k = 2 and k = 3
    findTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(2, MAX_PERIMETER, d_results, d_count);
    findTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(3, MAX_PERIMETER, d_results, d_count);

    // Launch kernel for k = 4 (equilateral triangles)
    findEquilateralTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(MAX_PERIMETER, d_results, d_count);

    cudaDeviceSynchronize();

    // Copy result count back to host
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (h_count > MAX_RESULTS) h_count = MAX_RESULTS;

    // Copy results back to host
    std::vector<Triangle> validTriangles(h_count);
    cudaMemcpy(validTriangles.data(), d_results, h_count * sizeof(Triangle), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_results);
    cudaFree(d_count);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    std::cout << "Found " << h_count << " valid triangles.\n";
    std::cout << "Time taken: " << elapsed.count() << " seconds.\n";

    // Optionally, print triangles
    /*
    for (const auto& triangle : validTriangles) {
        std::cout << "a = " << triangle.a << ", b = " << triangle.b << ", c = " << triangle.c << "\n";
    }
    */

    return 0;
}