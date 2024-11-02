#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

// Constants
constexpr uint64_t MAX_A_B_C_D = 90000;
constexpr uint64_t MAX_E = 120000; // Adjusted to ensure e^4 <= UINT64_MAX
constexpr uint64_t MIN_CHUNK_SIZE = 10;
constexpr uint64_t MAX_SOLUTIONS_PER_GPU = 2000000; // Adjust as needed

// Structure to hold a solution
struct Solution {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint64_t e;
};

// Structure to hold a^4 + b^4 sums
struct ABSum {
    uint64_t sum; // sum <= 1.3122e19 < UINT64_MAX
    uint64_t a;
    uint64_t b;
};

// CUDA Kernel to process assigned e's
__global__ void processEsKernel(
    const uint64_t* __restrict__ e_values, uint64_t num_es,
    const uint64_t* __restrict__ ab_sums, const uint64_t* __restrict__ ab_a, const uint64_t* __restrict__ ab_b, uint64_t ab_size,
    Solution* __restrict__ solutions,
    unsigned long long int* __restrict__ solution_count,
    uint64_t max_solutions // Boundary check
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < num_es; i += stride) {
        uint64_t e = e_values[i];
        uint64_t e4 = e * e * e * e;

        for (uint64_t c = 1; c <= MAX_A_B_C_D; c++) {
            uint64_t c4 = c * c * c * c;
            if (c4 >= e4) break;

            for (uint64_t d = c; d <= MAX_A_B_C_D; d++) { // Ensure c <= d
                uint64_t d4 = d * d * d * d;
                uint64_t cd4 = c4 + d4;
                if (cd4 >= e4) break; // Ensure c^4 + d^4 < e^4

                uint64_t remaining = e4 - cd4;

                // Binary search for remaining in ab_sums
                uint64_t left = 0;
                uint64_t right = ab_size;
                uint64_t mid = 0;
                bool found = false;
                uint64_t found_idx = 0;

                while (left < right) {
                    mid = left + (right - left) / 2;
                    uint64_t mid_val = ab_sums[mid];
                    if (mid_val < remaining) {
                        left = mid + 1;
                    }
                    else {
                        right = mid;
                    }
                }

                if (left < ab_size && ab_sums[left] == remaining) {
                    found = true;
                    found_idx = left;
                }

                if (found) {
                    uint64_t a = ab_a[found_idx];
                    uint64_t b = ab_b[found_idx];

                    // Ensure a <= b <= c to satisfy a <= b <= c <= d
                    if (b <= c) {
                        Solution sol;
                        sol.a = a;
                        sol.b = b;
                        sol.c = c;
                        sol.d = d;
                        sol.e = e;

                        // Atomic add with boundary check
                        unsigned long long int pos = atomicAdd(solution_count, 1ULL);
                        if (pos < max_solutions) {
                            solutions[pos] = sol;
                        }
                    }
                }
            }
        }
    }
}

// Function to initialize work queue with dynamically calculated chunk size
void initializeWorkQueue(uint64_t max_e, int num_threads, std::vector<std::pair<uint64_t, uint64_t>>& workChunks) {
    uint64_t chunk_size = std::max(max_e / static_cast<uint64_t>(num_threads * 10), static_cast<uint64_t>(MIN_CHUNK_SIZE));
    uint64_t current_e = 1;
    while (current_e <= max_e) {
        uint64_t end_e = std::min(current_e + chunk_size - 1, max_e);
        workChunks.emplace_back(std::make_pair(current_e, end_e));
        current_e += chunk_size;
    }
    std::cout << "Total work chunks: " << workChunks.size() << " with chunk size: " << chunk_size << std::endl;
}

// Function to precompute a^4 + b^4 sums
void precomputeABSums(const std::vector<uint64_t>& i_pows, std::vector<ABSum>& ab_pows_vec) {
    std::cout << "Starting precomputation of a^4 + b^4 sums..." << std::endl;

    // Reserve memory to prevent frequent reallocations
    ab_pows_vec.reserve(static_cast<size_t>(MAX_A_B_C_D) * (MAX_A_B_C_D + 1) / 2);

    for (uint64_t a = 1; a <= MAX_A_B_C_D; a++) {
        uint64_t a4 = i_pows[a];
        for (uint64_t b = a; b <= MAX_A_B_C_D; b++) { // Ensure a <= b
            uint64_t sum = a4 + i_pows[b];
            if (sum > i_pows[MAX_E]) break; // Do not store sums greater than e^4
            ab_pows_vec.emplace_back(ABSum{ sum, a, b });
        }
        if (a % 10000 == 0) {
            std::cout << "Precomputed a^4 + b^4 for a = " << a << std::endl;
        }
    }

    std::cout << "Precomputed all possible a^4 + b^4 sums." << std::endl;
}

int main() {
    // Start total computation time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Precompute i^4 using uint64_t
    std::vector<uint64_t> i_pows(MAX_E + 1, 0);
    for (uint64_t i = 1; i <= MAX_E; i++) {
        uint64_t i4 = i * i * i * i;
        // Overflow check
        if (i4 < i_pows[i]) {
            std::cerr << "Overflow detected for i = " << i << std::endl;
            return -1;
        }
        i_pows[i] = i4;
    }
    std::cout << "Precomputed all i^4 values up to e = " << MAX_E << "." << std::endl;

    // Precompute all possible a^4 + b^4 and store in a sorted vector
    std::vector<ABSum> ab_pows_vec;
    precomputeABSums(i_pows, ab_pows_vec);

    // Sort the ab_pows_vec based on sum to enable binary search
    std::cout << "Sorting the a^4 + b^4 sums for binary search..." << std::endl;
    std::sort(ab_pows_vec.begin(), ab_pows_vec.end(), [](const ABSum& lhs, const ABSum& rhs) -> bool {
        return lhs.sum < rhs.sum;
    });
    std::cout << "Sorted the a^4 + b^4 sums for binary search." << std::endl;

    // Split ab_pows_vec into separate arrays for sums, a, and b
    size_t ab_size = ab_pows_vec.size();
    std::vector<uint64_t> ab_sums_host(ab_size);
    std::vector<uint64_t> ab_a_host(ab_size);
    std::vector<uint64_t> ab_b_host(ab_size);

    for (size_t i = 0; i < ab_size; ++i) {
        ab_sums_host[i] = ab_pows_vec[i].sum;
        ab_a_host[i] = ab_pows_vec[i].a;
        ab_b_host[i] = ab_pows_vec[i].b;
    }

    // Free ab_pows_vec as it's no longer needed
    ab_pows_vec.clear();
    ab_pows_vec.shrink_to_fit();

    // Initialize work queue with dynamically calculated chunk size
    unsigned int numCPUs = std::thread::hardware_concurrency();
    if (numCPUs == 0) numCPUs = 4; // Fallback to 4 if unable to detect
    std::cout << "Number of CPU threads for work queue: " << numCPUs << std::endl;

    std::vector<std::pair<uint64_t, uint64_t>> workChunks; // Each chunk contains [e_start, e_end]
    initializeWorkQueue(MAX_E, numCPUs, workChunks);

    // Divide work chunks between GPUs: GPU0 gets 1/3 and GPU1 gets 2/3
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cerr << "This program requires at least two CUDA-capable GPUs." << std::endl;
        return -1;
    }
    std::cout << "Number of CUDA-capable GPUs detected: " << deviceCount << std::endl;

    // For simplicity, use first two GPUs
    int gpu0 = 0;
    int gpu1 = 1;

    size_t total_chunks = workChunks.size();
    size_t gpu0_chunks = total_chunks / 3;
    // Removed gpu1_chunks as it's unused

    std::vector<std::pair<uint64_t, uint64_t>> gpu0_work(workChunks.begin(), workChunks.begin() + gpu0_chunks);
    std::vector<std::pair<uint64_t, uint64_t>> gpu1_work(workChunks.begin() + gpu0_chunks, workChunks.end());

    // Function to process GPU work
    auto processGPU = [&](int device, const std::vector<std::pair<uint64_t, uint64_t>>& gpu_work, std::vector<Solution>& host_solutions, float& gpu_time) {
        // Set the current device
        cudaSetDevice(device);

        // Allocate device memory for ab_sums, ab_a, ab_b
        uint64_t* ab_sums_device;
        uint64_t* ab_a_device;
        uint64_t* ab_b_device;
        size_t ab_size_device = ab_size;

        cudaError_t err_code;

        // Allocate and copy ab_sums
        err_code = cudaMalloc(&ab_sums_device, ab_size_device * sizeof(uint64_t));
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA malloc failed for ab_sums_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            return;
        }
        err_code = cudaMemcpy(ab_sums_device, ab_sums_host.data(), ab_size_device * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for ab_sums_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            return;
        }

        // Allocate and copy ab_a
        err_code = cudaMalloc(&ab_a_device, ab_size_device * sizeof(uint64_t));
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA malloc failed for ab_a_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            return;
        }
        err_code = cudaMemcpy(ab_a_device, ab_a_host.data(), ab_size_device * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for ab_a_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            cudaFree(ab_a_device);
            return;
        }

        // Allocate and copy ab_b
        err_code = cudaMalloc(&ab_b_device, ab_size_device * sizeof(uint64_t));
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA malloc failed for ab_b_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            cudaFree(ab_a_device);
            return;
        }
        err_code = cudaMemcpy(ab_b_device, ab_b_host.data(), ab_size_device * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for ab_b_device on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            cudaFree(ab_a_device);
            cudaFree(ab_b_device);
            return;
        }

        // Allocate device memory for solutions
        Solution* d_solutions;
        err_code = cudaMalloc(&d_solutions, MAX_SOLUTIONS_PER_GPU * sizeof(Solution));
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA malloc failed for d_solutions on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            cudaFree(ab_a_device);
            cudaFree(ab_b_device);
            return;
        }
        cudaMemset(d_solutions, 0, MAX_SOLUTIONS_PER_GPU * sizeof(Solution));

        // Allocate device memory for solution count
        unsigned long long int* d_solution_count;
        err_code = cudaMalloc(&d_solution_count, sizeof(unsigned long long int));
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA malloc failed for d_solution_count on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            cudaFree(ab_sums_device);
            cudaFree(ab_a_device);
            cudaFree(ab_b_device);
            cudaFree(d_solutions);
            return;
        }
        cudaMemset(d_solution_count, 0, sizeof(unsigned long long int));

        // Allocate host memory for e_values and solutions per chunk
        std::vector<uint64_t> h_e_values;
        std::vector<Solution> h_solutions;
        h_solutions.reserve(MAX_SOLUTIONS_PER_GPU);

        // Start timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Iterate over work chunks assigned to this GPU
        for (const auto& chunk : gpu_work) {
            uint64_t e_start = chunk.first;
            uint64_t e_end = chunk.second;
            uint64_t num_es = e_end - e_start + 1;

            // Create a host array for e_values
            h_e_values.resize(num_es);
            for (uint64_t i = 0; i < num_es; i++) {
                h_e_values[i] = e_start + i;
            }

            // Allocate device memory for e_values
            uint64_t* d_e_values;
            err_code = cudaMalloc(&d_e_values, num_es * sizeof(uint64_t));
            if (err_code != cudaSuccess) {
                std::cerr << "CUDA malloc failed for d_e_values on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
                continue; // Skip this chunk
            }

            // Transfer e_values to device
            err_code = cudaMemcpy(d_e_values, h_e_values.data(), num_es * sizeof(uint64_t), cudaMemcpyHostToDevice);
            if (err_code != cudaSuccess) {
                std::cerr << "CUDA memcpy failed for d_e_values on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
                cudaFree(d_e_values);
                continue; // Skip this chunk
            }

            // Determine optimal block and grid sizes
            int blockSize = 256; // Must be multiple of 32
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            int gridSize = (num_es + blockSize - 1) / blockSize;
            gridSize = std::max(gridSize, static_cast<int>(deviceProp.multiProcessorCount * 32));

            // Launch the kernel
            processEsKernel<<<gridSize, blockSize>>>(
                d_e_values, num_es,
                ab_sums_device, ab_a_device, ab_b_device, ab_size_device,
                d_solutions, d_solution_count,
                MAX_SOLUTIONS_PER_GPU
            );

            // Check for kernel launch errors
            err_code = cudaGetLastError();
            if (err_code != cudaSuccess) {
                std::cerr << "Kernel launch failed on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
                cudaFree(d_e_values);
                continue; // Skip to next chunk
            }

            // Synchronize to ensure kernel completion
            cudaDeviceSynchronize();

            // Free e_values device memory
            cudaFree(d_e_values);
        }

        // Stop timing
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        gpu_time = milliseconds / 1000.0f; // Convert to seconds

        // Retrieve solution count
        unsigned long long int h_solution_count = 0;
        err_code = cudaMemcpy(&h_solution_count, d_solution_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        if (err_code != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for d_solution_count on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
        }

        // Allocate host memory for solutions
        if (h_solution_count > 0) {
            h_solutions.resize(h_solution_count);
            err_code = cudaMemcpy(h_solutions.data(), d_solutions, h_solution_count * sizeof(Solution), cudaMemcpyDeviceToHost);
            if (err_code != cudaSuccess) {
                std::cerr << "CUDA memcpy failed for d_solutions on device " << device << ": " << cudaGetErrorString(err_code) << std::endl;
            }
        }

        // Synchronize to ensure all copies are complete
        cudaDeviceSynchronize();

        // Append retrieved solutions to host_solutions
        if (h_solution_count > 0) {
            host_solutions.insert(host_solutions.end(), h_solutions.begin(), h_solutions.end());
        }

        // Cleanup
        cudaFree(ab_sums_device);
        cudaFree(ab_a_device);
        cudaFree(ab_b_device);
        cudaFree(d_solutions);
        cudaFree(d_solution_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    // Combined solutions from both GPUs
    std::vector<Solution> combinedSolutions;
    combinedSolutions.reserve(MAX_SOLUTIONS_PER_GPU * 2); // Assuming two GPUs

    // Variables to store GPU processing times
    float gpu0_time = 0.0f;
    float gpu1_time = 0.0f;

    // Launch threads to process each GPU
    std::thread gpu0_thread(processGPU, gpu0, gpu0_work, std::ref(combinedSolutions), std::ref(gpu0_time));
    std::thread gpu1_thread(processGPU, gpu1, gpu1_work, std::ref(combinedSolutions), std::ref(gpu1_time));

    // Wait for GPU threads to finish
    gpu0_thread.join();
    gpu1_thread.join();

    std::cout << "GPU0 processing time: " << gpu0_time << " seconds." << std::endl;
    std::cout << "GPU1 processing time: " << gpu1_time << " seconds." << std::endl;

    // Record total computation end time
    auto total_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> total_duration = total_stop - total_start;

    // Update totalSolutions based on combinedSolutions size
    uint64_t totalSolutions = combinedSolutions.size();

    // Display total solutions found
    std::cout << "Total solutions found: " << totalSolutions << std::endl;

    // Write solutions to a file
    std::ofstream outfile("solutions.txt");
    if (outfile.is_open()) {
        uint64_t solutionNumber = 1;
        for (const auto& sol : combinedSolutions) {
            outfile << "Solution " << solutionNumber << ": " 
                    << sol.a << "^4 + " 
                    << sol.b << "^4 + " 
                    << sol.c << "^4 + " 
                    << sol.d << "^4 = " 
                    << sol.e << "^4\n";
            solutionNumber++;
        }
        outfile.close();
        std::cout << "All solutions have been written to solutions.txt" << std::endl;
    } else {
        // If unable to open file, print to console (not recommended for large outputs)
        uint64_t solutionNumber = 1;
        for (const auto& sol : combinedSolutions) {
            std::cout << "Solution " << solutionNumber << ": " 
                      << sol.a << "^4 + " 
                      << sol.b << "^4 + " 
                      << sol.c << "^4 + " 
                      << sol.d << "^4 = " 
                      << sol.e << "^4" << std::endl;
            solutionNumber++;
        }
    }

    // Display computation times in seconds
    std::cout << "Total computation time: " << total_duration.count() << " seconds." << std::endl;
    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}