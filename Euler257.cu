#include <iostream>
#include <map>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_PERIMETER 100000000

// CUDA kernel for counting triangles (to be parallelized)
__global__ void countTriangles(int* d_validCount, int perimeter) {
    int a = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (a < perimeter / 3) {
        for (int b = a; b < (perimeter - a) / 2; ++b) {
            int c = perimeter - a - b;
            
            // Ensure triangle inequality holds
            if (c >= b && a + b > c && a + c > b && b + c > a) {
                // Calculate the ratio (a+b)(a+c)/bc using trial division
                int ab = a + b;
                int ac = a + c;
                int bc = b * c;

                // Check if (ab * ac) % (bc) == 0
                if ((ab * ac) % bc == 0) {
                    atomicAdd(d_validCount, 1);
                    // Debugging: Print the valid triangles
                    printf("Valid Triangle: a = %d, b = %d, c = %d\n", a, b, c);
                }
            }
        }
    }
}

int main() {
    int perimeter;
    std::cout << "Enter the perimeter limit: ";
    std::cin >> perimeter;

    if (perimeter < 3) {
        std::cout << "Invalid perimeter!" << std::endl;
        return 0;
    }

    int numTriangles = 0;

    // Allocate memory on the device
    int* d_validCount;
    cudaMalloc(&d_validCount, sizeof(int));
    cudaMemset(d_validCount, 0, sizeof(int));

    // Set up timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int numBlocks = (perimeter / 3 + threadsPerBlock - 1) / threadsPerBlock;
    countTriangles<<<numBlocks, threadsPerBlock>>>(d_validCount, perimeter);

    // Synchronize to wait for all threads to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&numTriangles, d_validCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_validCount);

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Number of valid triangles: " << numTriangles << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    return 0;
}