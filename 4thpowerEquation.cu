#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>

// Constants
constexpr uint64_t MAX_A_B_C_D = 50000;
constexpr uint64_t MAX_E = 65535; // Adjusted to fit within uint64_t
constexpr __int128 MAX_E4 = (__int128)MAX_E * MAX_E * MAX_E * MAX_E;

// Structure to hold a partial solution (c, d, e)
struct PartialSolution {
    uint64_t c;
    uint64_t d;
    uint64_t e;
};

// Structure to hold a complete solution (a, b, c, d, e)
struct Solution {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint64_t e;
};

// Error checking macro
#define CUDA_CHECK_ERROR(call)                                                 \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Device function for binary search
__device__ bool binary_search_device(const uint64_t* array, size_t size, uint64_t target) {
    size_t left = 0;
    size_t right = size;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (array[mid] == target) {
            return true;
        } else if (array[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return false;
}

// CUDA Kernel to find partial solutions
__global__ void findSolutionsKernel(
    const uint64_t* a4b4_sums, size_t a4b4_size,
    const uint64_t* i_pows,
    uint64_t e_start, uint64_t e_end,
    PartialSolution* partial_solutions,
    unsigned long long int* partial_count)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t e = e_start + idx;
    if (e > e_end || e > 65535) return;

    uint64_t e4 = i_pows[e];

    // Iterate over c
    for (uint64_t c = 1; c <= MAX_A_B_C_D; ++c) {
        uint64_t c4 = i_pows[c];
        if (c4 >= e4) break; // c^4 >= e^4

        // Iterate over d starting from c to ensure c <= d
        for (uint64_t d = c; d <= MAX_A_B_C_D; ++d) {
            uint64_t d4 = i_pows[d];
            uint64_t cd4 = c4 + d4;
            if (cd4 >= e4) break; // c^4 + d^4 >= e^4

            uint64_t remaining = e4 - cd4;

            // Perform binary search for remaining in a4b4_sums
            bool found = binary_search_device(a4b4_sums, a4b4_size, remaining);
            if (found) {
                // Atomic increment to get unique index
                unsigned long long int sol_idx = atomicAdd(partial_count, 1);
                if (sol_idx < 10000000) { // Ensure we don't exceed allocated memory
                    partial_solutions[sol_idx].c = c;
                    partial_solutions[sol_idx].d = d;
                    partial_solutions[sol_idx].e = e;
                }
            }
        }
    }
}

// Host function to precompute i^4
std::vector<uint64_t> precompute_powers(uint64_t max_val) {
    std::vector<uint64_t> i_pows(max_val + 1, 0);
    for (uint64_t i = 1; i <= max_val; ++i) {
        i_pows[i] = i * i * i * i; // i^4
    }
    return i_pows;
}

// Multithreaded Host function to precompute a^4 + b^4 sums and build a sum-to-(a,b) map
std::vector<uint64_t> precompute_a4b4_sums_multithreaded(
    const std::vector<uint64_t>& i_pows,
    uint64_t max_a_b_c_d,
    uint64_t max_e4,
    std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>& sum_to_ab_map)
{
    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 threads if unable to detect
    std::cout << "Precomputing a^4 + b^4 sums using " << num_threads << " threads...\n";

    // Calculate the range of 'a' for each thread
    uint64_t a_per_thread = max_a_b_c_d / num_threads;
    uint64_t remaining_a = max_a_b_c_d % num_threads;

    // Vectors to hold per-thread results
    std::vector<std::vector<uint64_t>> thread_sums(num_threads);
    std::vector<std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>> thread_maps(num_threads);

    // Lambda function for each thread's work
    auto worker = [&](unsigned int thread_id, uint64_t a_start, uint64_t a_end) {
        std::vector<uint64_t>& local_sums = thread_sums[thread_id];
        std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>& local_map = thread_maps[thread_id];
        local_sums.reserve((a_end - a_start + 1) * (max_a_b_c_d - a_start + 1) / 2); // Approximate reserve

        for (uint64_t a = a_start; a <= a_end; ++a) {
            uint64_t a4 = i_pows[a];
            for (uint64_t b = a; b <= max_a_b_c_d; ++b) { // Ensure a <= b
                uint64_t sum = a4 + i_pows[b];
                if (sum > max_e4) break; // Do not store sums greater than e^4
                local_sums.push_back(sum);
                // Build the sum-to-(a,b) map for memoization
                local_map[sum].emplace_back(a, b);
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    uint64_t current_a = 1;
    for (unsigned int t = 0; t < num_threads; ++t) {
        uint64_t a_start = current_a;
        uint64_t a_end = a_start + a_per_thread - 1;
        if (t == num_threads - 1) {
            a_end += remaining_a; // Add any remaining 'a' to the last thread
        }
        threads.emplace_back(worker, t, a_start, a_end);
        current_a = a_end + 1;
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Merge results from all threads
    std::vector<uint64_t> a4b4_sums;
    a4b4_sums.reserve(max_a_b_c_d * max_a_b_c_d / 2); // Adjust as needed

    for (unsigned int t = 0; t < num_threads; ++t) {
        // Merge sums
        a4b4_sums.insert(a4b4_sums.end(), thread_sums[t].begin(), thread_sums[t].end());

        // Merge maps
        for (const auto& pair : thread_maps[t]) {
            sum_to_ab_map[pair.first].insert(
                sum_to_ab_map[pair.first].end(),
                pair.second.begin(),
                pair.second.end()
            );
        }
    }

    // Sort the sums for binary search
    std::sort(a4b4_sums.begin(), a4b4_sums.end());

    return a4b4_sums;
}

// Function to launch kernels on a specific GPU with automatic block and grid size determination
void process_on_gpu(
    int device_id,
    const std::vector<uint64_t>& a4b4_sums,
    const std::vector<uint64_t>& i_pows,
    uint64_t e_start,
    uint64_t e_end,
    std::vector<PartialSolution>& host_partial_solutions,
    float& gpu_time)
{
    CUDA_CHECK_ERROR(cudaSetDevice(device_id));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    // Allocate and copy a4b4_sums to device
    uint64_t* d_a4b4_sums;
    size_t a4b4_bytes = a4b4_sums.size() * sizeof(uint64_t);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_a4b4_sums, a4b4_bytes));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(d_a4b4_sums, a4b4_sums.data(), a4b4_bytes, cudaMemcpyHostToDevice, stream));

    // Allocate and copy i_pows to device
    uint64_t* d_i_pows;
    size_t i_pows_bytes = i_pows.size() * sizeof(uint64_t);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_i_pows, i_pows_bytes));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(d_i_pows, i_pows.data(), i_pows_bytes, cudaMemcpyHostToDevice, stream));

    // Estimate maximum number of partial solutions
    size_t max_partial_solutions = 10000000; // Adjust as needed
    PartialSolution* d_partial_solutions;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_partial_solutions, max_partial_solutions * sizeof(PartialSolution)));
    unsigned long long int* d_partial_count;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_partial_count, sizeof(unsigned long long int)));
    CUDA_CHECK_ERROR(cudaMemsetAsync(d_partial_count, 0, sizeof(unsigned long long int), stream));

    // Determine optimal block and grid sizes using cudaOccupancyMaxPotentialBlockSize
    int block_size;
    int grid_size;
    CUDA_CHECK_ERROR(cudaOccupancyMaxPotentialBlockSize(
        &grid_size,
        &block_size,
        findSolutionsKernel,
        0, // Dynamic shared memory size
        0  // Block size limit
    ));

    // Calculate the number of blocks needed based on the e range
    uint64_t total_e = e_end - e_start + 1;
    int threadsPerBlock = block_size;
    int blocksPerGrid = (total_e + threadsPerBlock - 1) / threadsPerBlock;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    findSolutionsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_a4b4_sums, a4b4_sums.size(),
        d_i_pows,
        e_start, e_end,
        d_partial_solutions,
        d_partial_count
    );

    // Check for kernel launch errors
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = stop - start;
    gpu_time += duration.count();

    // Retrieve partial solution count
    unsigned long long int h_partial_count = 0;
    CUDA_CHECK_ERROR(cudaMemcpyAsync(&h_partial_count, d_partial_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // Retrieve partial solutions
    std::vector<PartialSolution> h_partial_solutions(h_partial_count);
    CUDA_CHECK_ERROR(cudaMemcpyAsync(h_partial_solutions.data(), d_partial_solutions, h_partial_count * sizeof(PartialSolution), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // Append to host partial solutions
    host_partial_solutions.insert(host_partial_solutions.end(), h_partial_solutions.begin(), h_partial_solutions.end());

    // Cleanup
    CUDA_CHECK_ERROR(cudaFree(d_a4b4_sums));
    CUDA_CHECK_ERROR(cudaFree(d_i_pows));
    CUDA_CHECK_ERROR(cudaFree(d_partial_solutions));
    CUDA_CHECK_ERROR(cudaFree(d_partial_count));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}

int main() {
    // Start total computation time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Precompute i^4 on host
    std::cout << "Precomputing i^4...\n";
    std::vector<uint64_t> i_pows = precompute_powers(MAX_E);
    std::cout << "Precomputed i^4 up to " << MAX_E << ".\n";

    // Precompute a^4 + b^4 sums on host and build sum-to-(a,b) map for memoization using multithreading
    std::cout << "Precomputing a^4 + b^4 sums and building sum-to-(a,b) map using multithreading...\n";
    std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> sum_to_ab_map;
    std::vector<uint64_t> a4b4_sums = precompute_a4b4_sums_multithreaded(i_pows, MAX_A_B_C_D, MAX_E4, sum_to_ab_map);
    std::cout << "Precomputed and sorted a^4 + b^4 sums. Total sums: " << a4b4_sums.size() << "\n";

    // Determine the number of CUDA-capable devices
    int device_count = 0;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        std::cerr << "This program requires at least one CUDA-capable GPU.\n";
        return EXIT_FAILURE;
    }
    std::cout << "Number of CUDA-capable devices detected: " << device_count << "\n";

    // Define e ranges dynamically based on device count
    uint64_t e_total = MAX_E;
    uint64_t e_per_gpu = e_total / device_count;

    // Prepare e ranges for each GPU
    std::vector<std::pair<uint64_t, uint64_t>> gpu_e_ranges;
    for (int i = 0; i < device_count; ++i) {
        uint64_t e_start = i * e_per_gpu + 1;
        uint64_t e_end = (i == device_count - 1) ? MAX_E : (e_start + e_per_gpu - 1);
        gpu_e_ranges.emplace_back(e_start, e_end);
    }

    // Display e ranges for each GPU (for debugging)
    for (int i = 0; i < device_count; ++i) {
        std::cout << "GPU " << i << " assigned e range: " << gpu_e_ranges[i].first << " to " << gpu_e_ranges[i].second << "\n";
    }

    // Vectors to hold partial solutions from all GPUs
    std::vector<std::vector<PartialSolution>> all_partial_solutions(device_count);
    // Timers for each GPU
    std::vector<float> gpu_times(device_count, 0.0f);

    // Launch processing on all GPUs using separate host threads
    std::vector<std::thread> gpu_threads;
    for (int i = 0; i < device_count; ++i) {
        gpu_threads.emplace_back(process_on_gpu, i, std::cref(a4b4_sums), std::cref(i_pows),
                                 gpu_e_ranges[i].first, gpu_e_ranges[i].second,
                                 std::ref(all_partial_solutions[i]),
                                 std::ref(gpu_times[i]));
    }

    // Wait for all threads to finish
    for (auto& th : gpu_threads) {
        th.join();
    }

    // Aggregate all partial solutions
    std::vector<PartialSolution> aggregated_partial_solutions;
    for (const auto& vec : all_partial_solutions) {
        aggregated_partial_solutions.insert(aggregated_partial_solutions.end(), vec.begin(), vec.end());
    }
    std::cout << "Total partial solutions found: " << aggregated_partial_solutions.size() << "\n";

    // Iterate through all partial solutions and find corresponding a and b
    std::vector<Solution> complete_solutions;
    complete_solutions.reserve(aggregated_partial_solutions.size() * 2); // Estimate based on average (a,b) per sum

    for (const auto& partial : aggregated_partial_solutions) {
        uint64_t remaining = i_pows[partial.e] - i_pows[partial.c] - i_pows[partial.d];
        auto it = sum_to_ab_map.find(remaining);
        if (it != sum_to_ab_map.end()) {
            // For each (a, b) pair that sums to 'remaining', create a complete solution
            for (const auto& ab_pair : it->second) {
                // Enforce a <= b <= c <= d
                if (ab_pair.second <= partial.c) { // Ensure b <= c
                    Solution sol;
                    sol.a = ab_pair.first;
                    sol.b = ab_pair.second;
                    sol.c = partial.c;
                    sol.d = partial.d;
                    sol.e = partial.e;
                    complete_solutions.push_back(sol);
                }
            }
        }
    }

    // Sort the complete solutions based on e, c, d, a, b
    std::sort(complete_solutions.begin(), complete_solutions.end(),
        [](const Solution& a, const Solution& b) -> bool {
            if (a.e != b.e)
                return a.e < b.e;
            if (a.c != b.c)
                return a.c < b.c;
            if (a.d != b.d)
                return a.d < b.d;
            if (a.a != b.a)
                return a.a < b.a;
            return a.b < b.b;
        });

    // Display results
    uint64_t totalSolutions = complete_solutions.size();
    std::cout << "\nTotal complete solutions found: " << totalSolutions << "\n";

    uint64_t solutionNumber = 1;
    for (const auto& sol : complete_solutions) {
        std::cout << "Solution " << solutionNumber << ": " 
                  << sol.a << "^4 + " 
                  << sol.b << "^4 + " 
                  << sol.c << "^4 + " 
                  << sol.d << "^4 = " 
                  << sol.e << "^4\n";
        solutionNumber++;
    }

    // Display computation times
    for (int i = 0; i < device_count; ++i) {
        std::cout << "\nGPU " << i << " (Device " << i << ") computation time: " << gpu_times[i] << " seconds.\n";
    }

    // Total computation time
    auto total_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> total_duration = total_stop - total_start;
    std::cout << "Total computation time: " << total_duration.count() << " seconds.\n";

    std::cout << "Computation completed successfully.\n";
    return 0;
}