#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <atomic>

#define MAX_PERIMETER 10000000

// Atomic variable to count triangles
__device__ atomicInt64_t triangle_count;
__device__ int64_t triangle_data[1000000]; // Store every millionth triangle

__device__ int64_t gcd(int64_t a, int64_t b) {
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

// Kernel to count primitive triangles
__global__ void count_triangles(int64_t max_perimeter, int64_t start, int64_t end) {
    int64_t p = blockIdx.x * blockDim.x + threadIdx.x + start;

    if (p < end) {
        for (int64_t a = 1; a <= p / 3; ++a) {
            for (int64_t b = a; b <= (p - a) / 2; ++b) {
                int64_t c = p - a - b;
                if (c >= b && a + b > c) { // Check triangle inequality
                    if (gcd(gcd(a, b), c) == 1) { // Check if primitive
                        int64_t count = atomicAdd(&triangle_count, 1); // Atomic increment
                        if ((count + 1) % 1000000 == 0) { // Save every millionth triangle
                            triangle_data[(count + 1) / 1000000 - 1] = (a << 40) | (b << 20) | c; // Store (a, b, c) in a single int64_t
                        }
                    }
                }
            }
        }
    }
}

// Function to automatically determine optimal block and grid sizes
void getOptimalBlockAndGridSize(int64_t totalWork, dim3& gridSize, dim3& blockSize) {
    int deviceId = 0;
    cudaSetDevice(deviceId);
    
    // Set up initial variables for occupancy calculation
    int minGridSize;
    int blockSizeMax;

    // Get the maximum potential block size and grid size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax, count_triangles, 0, 0);

    // Set the determined block size
    blockSize.x = blockSizeMax;

    // Calculate grid size
    long long numBlocks = (totalWork + blockSize.x - 1) / blockSize.x;
    gridSize.x = static_cast<int>(numBlocks);
}

int main() {
    // Initialize device variables
    int64_t h_triangle_count = 0;
    cudaMemcpyToSymbol(triangle_count, &h_triangle_count, sizeof(int64_t));

    // Get CUDA device properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using device: " << deviceProp.name << std::endl;

    // Automatically determine block and grid size
    dim3 blockSize;
    dim3 gridSize;
    getOptimalBlockAndGridSize(MAX_PERIMETER, gridSize, blockSize);

    // Create CUDA streams
    const int numStreams = 2; // Number of streams to use
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch kernels in streams
    for (int i = 0; i < numStreams; ++i) {
        int64_t range_start = (MAX_PERIMETER / numStreams) * i + 1;
        int64_t range_end = (i == numStreams - 1) ? MAX_PERIMETER + 1 : (MAX_PERIMETER / numStreams) * (i + 1) + 1;

        count_triangles<<<gridSize, blockSize, 0, streams[i]>>>(MAX_PERIMETER, range_start, range_end);
    }

    // Wait for all streams to complete
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy result back to host
    cudaMemcpyFromSymbol(&h_triangle_count, triangle_count, sizeof(int64_t));

    // Copy triangle data back to host
    int64_t h_triangle_data[1000000];
    cudaMemcpy(h_triangle_data, triangle_data, sizeof(h_triangle_data));

    // Output every millionth triangle
    for (int i = 0; i < (h_triangle_count + 1) / 1000000; ++i) {
        int64_t triangle = h_triangle_data[i];
        int64_t a = (triangle >> 40) & 0xFFFFF;
        int64_t b = (triangle >> 20) & 0xFFFFF;
        int64_t c = triangle & 0xFFFFF;
        std::cout << "Found triangle #" << (i + 1) * 1000000 << ": (" << a << ", " << b << ", " << c << ")\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Total primitive triangles found: " << h_triangle_count << "\n";
    std::cout << "Execution time: " << duration.count() << " seconds.\n";

    // Clean up streams
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}