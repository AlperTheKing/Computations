#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

// Constants
constexpr int MAX_PERIMETER = 10'000'000;
constexpr int MAX_A = MAX_PERIMETER / 3;
constexpr int PROGRESS_STEP = 1'000'000; // Progress step for monitoring

// CUDA-compatible GCD function
__device__ int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// CUDA kernel to calculate primitive triangles
__global__ void find_primitive_triangles(int start_a, int end_a, int max_perimeter, unsigned long long int* triangle_count, int* progress, int device_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int a = start_a + idx; a <= end_a; a += stride) {
        for (int b = a; a + b < max_perimeter; ++b) {
            int c = a + b - 1; // Initial c value
            while (a + b + c <= max_perimeter) {
                int perimeter = a + b + c;

                // Check if it forms a valid triangle and is primitive
                if ((a + b > c) && (gcd(gcd(a, b), c) == 1)) {
                    atomicAdd(triangle_count, 1ULL);
                }

                // Update progress
                if (perimeter / PROGRESS_STEP > *progress) {
                    atomicMax(progress, perimeter / PROGRESS_STEP);
                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        printf("Device %d - Progress: Reached perimeter %d, Current count: %llu\n", device_id, perimeter, *triangle_count);
                    }
                }

                c++;
            }
        }
    }
}

// Check for CUDA errors
#define cudaCheckError(call) {                                           \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";  \
        exit(err);                                                       \
    }                                                                    \
}

// Function to launch kernel on a specific device
void launch_on_device(int device_id, int start_a, int end_a, unsigned long long int& triangle_count) {
    cudaCheckError(cudaSetDevice(device_id));

    int block_size;       // The optimal number of threads per block
    int min_grid_size;    // The minimum grid size needed to achieve maximum occupancy
    cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, find_primitive_triangles, 0, 0));

    int grid_size = (end_a - start_a + 1 + block_size - 1) / block_size;

    std::cout << "Device " << device_id << " - Optimal block size: " << block_size
              << ", Grid size: " << grid_size << "\n";
    std::cout << "Device " << device_id << " - Processing range: [" << start_a << ", " << end_a << "]\n";

    unsigned long long int* d_triangle_count;
    int* d_progress;
    
    // Create CUDA stream for this device
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));

    cudaCheckError(cudaMalloc(&d_triangle_count, sizeof(unsigned long long int)));
    cudaCheckError(cudaMalloc(&d_progress, sizeof(int)));

    cudaCheckError(cudaMemcpyAsync(d_triangle_count, &triangle_count, sizeof(unsigned long long int), cudaMemcpyHostToDevice, stream));
    cudaCheckError(cudaMemsetAsync(d_progress, 0, sizeof(int), stream));

    // Launch kernel with calculated grid and block sizes
    find_primitive_triangles<<<grid_size, block_size, 0, stream>>>(start_a, end_a, MAX_PERIMETER, d_triangle_count, d_progress, device_id);

    // Copy result back to host
    cudaCheckError(cudaMemcpyAsync(&triangle_count, d_triangle_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost, stream));

    cudaCheckError(cudaStreamSynchronize(stream));
    cudaCheckError(cudaStreamDestroy(stream));
    cudaCheckError(cudaFree(d_triangle_count));
    cudaCheckError(cudaFree(d_progress));
}

// Main function to handle multi-GPU execution
void calculate_triangles_with_cuda() {
    int num_devices;
    cudaCheckError(cudaGetDeviceCount(&num_devices));

    if (num_devices < 1) {
        std::cerr << "No CUDA-capable device detected.\n";
        return;
    }

    std::vector<unsigned long long int> triangle_counts(num_devices, 0);
    unsigned long long int total_triangle_count = 0;

    int workload_per_device = MAX_A / num_devices;
    std::vector<std::thread> threads;

    std::cout << "Number of detected CUDA-capable devices: " << num_devices << "\n";

    for (int device_id = 0; device_id < num_devices; ++device_id) {
        int start_a = device_id * workload_per_device + 1;
        int end_a = (device_id == num_devices - 1) ? MAX_A : start_a + workload_per_device - 1;

        // Launch each device workload in a separate CPU thread
        threads.emplace_back(launch_on_device, device_id, start_a, end_a, std::ref(triangle_counts[device_id]));
    }

    // Join all threads
    for (auto& t : threads) {
        t.join();
    }

    // Aggregate results from each GPU
    for (int device_id = 0; device_id < num_devices; ++device_id) {
        total_triangle_count += triangle_counts[device_id];
        std::cout << "Device " << device_id << " - Partial triangle count: " << triangle_counts[device_id] << "\n";
    }

    std::cout << "Total number of primitive triangles: " << total_triangle_count << "\n";
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    calculate_triangles_with_cuda();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}