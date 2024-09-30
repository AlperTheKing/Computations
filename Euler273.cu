#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <fstream>

// Structure to store pairs of (a, b) with the corresponding N
struct Solution {
    long long N;
    int a;
    int b;
};

// CUDA kernel to compute sum of two squares for a given N
__global__ void sum_of_two_squares(long long* Ns, int* a_results, int* b_results, int num_Ns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_Ns) return;

    long long N = Ns[idx];
    for (int a = 1; a * a <= N; ++a) {
        long long b_squared = N - static_cast<long long>(a) * a;
        int b = static_cast<int>(sqrtf(b_squared));
        if (b * b == b_squared) {
            a_results[idx] = a;
            b_results[idx] = b;
            return;
        }
    }
    a_results[idx] = -1;  // No solution found
    b_results[idx] = -1;
}

// Function to generate primes of the form 4k + 1 using sieve of Eratosthenes
std::vector<int> sieve_of_eratosthenes(int limit) {
    std::vector<bool> is_prime(limit + 1, true);
    std::vector<int> primes;
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= limit; ++i) {
        if (is_prime[i]) {
            if (i % 4 == 1) {
                primes.push_back(i);
            }
            for (int j = 2 * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return primes;
}

// Generate all non-empty combinations of primes and calculate the product
void generate_prime_combinations(const std::vector<int>& primes, std::vector<long long>& Ns) {
    size_t num_primes = primes.size();
    for (size_t i = 1; i < (1 << num_primes); ++i) {
        long long N = 1;
        for (size_t j = 0; j < num_primes; ++j) {
            if (i & (1 << j)) {
                N *= primes[j];
            }
        }
        Ns.push_back(N);
    }
}

// Helper function to print long long values as string
std::string long_long_to_string(long long value) {
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

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Limit for primes of the form 4k + 1
    int prime_limit = 150;

    // Generate primes of the form 4k + 1 using sieve
    std::vector<int> primes = sieve_of_eratosthenes(prime_limit);

    // Generate all non-empty combinations of primes
    std::vector<long long> Ns;
    generate_prime_combinations(primes, Ns);
    int num_Ns = Ns.size();

    // Allocate memory for results (a and b values)
    long long* d_Ns;
    int* d_a_results;
    int* d_b_results;
    int* a_results = new int[num_Ns];
    int* b_results = new int[num_Ns];

    cudaMalloc(&d_Ns, num_Ns * sizeof(long long));
    cudaMalloc(&d_a_results, num_Ns * sizeof(int));
    cudaMalloc(&d_b_results, num_Ns * sizeof(int));

    cudaMemcpy(d_Ns, Ns.data(), num_Ns * sizeof(long long), cudaMemcpyHostToDevice);

    // Determine optimal block and grid sizes using cudaOccupancyMaxPotentialBlockSize
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sum_of_two_squares, 0, num_Ns);
    int gridSize = (num_Ns + blockSize - 1) / blockSize;

    // Launch the kernel
    sum_of_two_squares<<<gridSize, blockSize>>>(d_Ns, d_a_results, d_b_results, num_Ns);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(a_results, d_a_results, num_Ns * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_results, d_b_results, num_Ns * sizeof(int), cudaMemcpyDeviceToHost);

    // Store solutions for square-free N
    std::vector<Solution> solutions;

    for (int i = 0; i < num_Ns; ++i) {
        if (a_results[i] != -1 && b_results[i] != -1) {
            solutions.push_back({Ns[i], a_results[i], b_results[i]});
        }
    }

    // Sort the solutions by N in ascending order
    std::sort(solutions.begin(), solutions.end(), [](const Solution& s1, const Solution& s2) {
        return s1.N < s2.N;
    });

    // Output to file
    std::ofstream file("SumofSquares.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file SumofSquares.txt for writing." << std::endl;
        return 1;
    }

    long long total_sum_of_a = 0;

    // Write sorted solutions to the file and calculate the total sum of a's
    for (const auto& sol : solutions) {
        file << "N = " << long_long_to_string(sol.N) << ", a = " << sol.a << ", b = " << sol.b << std::endl;
        total_sum_of_a += sol.a;  // Accumulate the sum of a's
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Write the total sum of a's and the elapsed time to the file
    file << "Total sum of a values: " << long_long_to_string(total_sum_of_a) << std::endl;
    file << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // Close the file
    file.close();

    // Free CUDA memory
    cudaFree(d_Ns);
    cudaFree(d_a_results);
    cudaFree(d_b_results);

    // Output results to console
    std::cout << "Found " << solutions.size() << " solutions in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Total sum of a values: " << total_sum_of_a << std::endl;

    delete[] a_results;
    delete[] b_results;

    return 0;
}