#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>
#include <set>
#include <map>
#include <algorithm>
#include <string>
#include <cstring>

// Declare primes in host and __constant__ for device
int h_primes[] = {5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149};
__constant__ int primes[16];  // Max 16 primes
const int num_primes = sizeof(h_primes) / sizeof(h_primes[0]);

// Bitwise method to compute integer square root of a non-negative integer n
__device__ int int_sqrt(int n) {
    int res = 0;
    int bit = 1 << 30; // Largest power of 4 less than the maximum integer (2^30)

    // Find the largest bit that's smaller or equal to the number
    while (bit > n) {
        bit >>= 2;
    }

    // Compute the integer square root
    while (bit != 0) {
        if (n >= res + bit) {
            n -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }

    return res;
}

// CUDA kernel for finding Gaussian integer pairs (a, b) for a given prime p
__global__ void find_gaussian_pairs(int p, int* a_results, int* b_results, int* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= p) return;

    int a = idx;
    int b_squared = p - a * a;
    if (b_squared < 0) return;

    int b = int_sqrt(b_squared);  // Use bitwise integer square root
    if (b * b == b_squared && a <= b) {
        int pos = atomicAdd(count, 1);  // Atomic addition to get a unique position
        a_results[pos] = a;
        b_results[pos] = b;
    }
}

// Custom function to convert __int128 to string
std::string int128_to_string(__int128 value) {
    if (value == 0) return "0";
    std::string result;
    bool negative = value < 0;
    if (negative) value = -value;
    while (value > 0) {
        result.insert(result.begin(), '0' + (value % 10));
        value /= 10;
    }
    if (negative) result.insert(result.begin(), '-');
    return result;
}

// Function to combine two sets of Gaussian integer pairs (a1, b1) and (a2, b2)
// Ensures that 0 <= a <= b for each resulting pair
std::set<std::pair<int, int>> combine_gaussian_pairs(const std::set<std::pair<int, int>>& pairs1,
                                                     const std::set<std::pair<int, int>>& pairs2) {
    std::set<std::pair<int, int>> result;
    for (const auto& [a1, b1] : pairs1) {
        for (const auto& [a2, b2] : pairs2) {
            // Consider both (a1*a2 - b1*b2, a1*b2 + a2*b1) and (a1*a2 + b1*b2, a1*b2 - a2*b1) combinations
            int a_plus = abs(a1 * a2 - b1 * b2);
            int b_plus = abs(a1 * b2 + a2 * b1);
            result.emplace(std::min(a_plus, b_plus), std::max(a_plus, b_plus));  // Ensure 0 <= a <= b

            int a_minus = abs(a1 * a2 + b1 * b2);
            int b_minus = abs(a1 * b2 - a2 * b1);
            result.emplace(std::min(a_minus, b_minus), std::max(a_minus, b_minus));  // Ensure 0 <= a <= b
        }
    }
    return result;
}

// Host function to find Gaussian integer pairs for a prime p
std::vector<std::pair<int, int>> find_gaussian_pairs_host(int p, int* d_a_results, int* d_b_results, int* d_count) {
    std::vector<std::pair<int, int>> pairs;

    int count = 0;
    int max_pairs = p;  // Max possible pairs can't exceed p

    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    // Determine optimal block and grid sizes
    int blockSize = 0; // Optimal block size
    int minGridSize = 0; // Minimum grid size to achieve maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, find_gaussian_pairs, 0, p);

    int gridSize = (p + blockSize - 1) / blockSize;

    // Launch the kernel
    find_gaussian_pairs<<<gridSize, blockSize>>>(p, d_a_results, d_b_results, d_count);
    cudaDeviceSynchronize();

    // Copy the results back to host
    int* a_results = new int[max_pairs];
    int* b_results = new int[max_pairs];
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_results, d_a_results, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_results, d_b_results, count * sizeof(int), cudaMemcpyDeviceToHost);

    // Store the results in pairs
    for (int i = 0; i < count; ++i) {
        if (a_results[i] <= b_results[i]) {  // Ensure 0 <= a <= b
            pairs.emplace_back(a_results[i], b_results[i]);
        }
    }

    // Clean up
    delete[] a_results;
    delete[] b_results;

    return pairs;
}

// Host function to generate square-free N values and find Gaussian integer solutions
void generate_gaussian_solutions(std::ofstream& output_file, __int128& total_sum_of_a, 
                                 int* d_a_results, int* d_b_results, int* d_count) {
    for (int mask = 1; mask < (1 << num_primes); ++mask) {
        __int128 N = 1;
        std::set<std::pair<int, int>> combined_pairs;
        bool first = true;
        std::string factors;

        for (size_t i = 0; i < num_primes; ++i) {
            if (mask & (1 << i)) {
                N *= h_primes[i];  // Use host-side primes array
                factors += (factors.empty() ? "" : "*") + std::to_string(h_primes[i]);
                auto prime_pairs = find_gaussian_pairs_host(h_primes[i], d_a_results, d_b_results, d_count);
                std::set<std::pair<int, int>> prime_pair_set(prime_pairs.begin(), prime_pairs.end());

                if (first) {
                    combined_pairs = prime_pair_set;
                    first = false;
                } else {
                    combined_pairs = combine_gaussian_pairs(combined_pairs, prime_pair_set);
                }
            }
        }

        if (!combined_pairs.empty()) {
            output_file << "N = " << int128_to_string(N) << " (" << factors << "): ";
            for (const auto& [a, b] : combined_pairs) {
                output_file << "(" << a << "," << b << ") ";
                total_sum_of_a += a;  // Accumulate sum of a values
            }
            output_file << std::endl;
        }
    }
}

int main() {
    // Copy primes to device constant memory
    cudaMemcpyToSymbol(primes, h_primes, sizeof(h_primes));

    // Measure overall execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate memory on device (reuse this across multiple kernel calls)
    int* d_a_results;
    int* d_b_results;
    int* d_count;
    int max_pairs = 149;  // Max prime value in the list
    cudaMalloc(&d_a_results, max_pairs * sizeof(int));
    cudaMalloc(&d_b_results, max_pairs * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    // Open the output file to write the results (SumofSquares.txt)
    std::ofstream output_file("SumofSquares.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open file SumofSquares.txt for writing." << std::endl;
        return 1;
    }

    // Variable to store the total sum of all 'a' values
    __int128 total_sum_of_a = 0;

    // Measure time for generating and finding Gaussian integer pairs
    auto computation_start = std::chrono::high_resolution_clock::now();
    generate_gaussian_solutions(output_file, total_sum_of_a, d_a_results, d_b_results, d_count);
    auto computation_end = std::chrono::high_resolution_clock::now();

    // Close the file
    output_file.close();

    // Free device memory
    cudaFree(d_a_results);
    cudaFree(d_b_results);
    cudaFree(d_count);

    // Compute the elapsed time for computation
    std::chrono::duration<double> computation_duration = computation_end - computation_start;

    // Print the computation time
    std::cout << "Computation time: " << computation_duration.count() << " seconds" << std::endl;

    // Print the total sum of all 'a' values
    std::cout << "Total sum of all 'a' values: " << int128_to_string(total_sum_of_a) << std::endl;

    return 0;
}