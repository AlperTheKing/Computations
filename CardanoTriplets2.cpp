#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

// Function to count valid triplets (a, b, c)
long long count_triplets(long long start_k, long long end_k, long long max_sum) {
    long long total_count = 0;

    for (long long k = start_k; k <= end_k; k += 1) {
        long long a = 3 * k + 2;  // a = 3k + 2
        long long term = (k + 1) * (k + 1) * (8 * k + 5);  // b^2 * c = (k + 1)^2 * (8k + 5)

        for (long long b = 1; b * b <= term; ++b) {
            if (term % (b * b) == 0) {  // Check if b^2 divides term
                long long c = term / (b * b);  // Calculate c

                // Check if the sum a + b + c is within the allowed limit
                if (a + b + c <= max_sum) {
                    total_count++;
                }
            }
        }
    }

    return total_count;
}

int main() {
    const long long max_sum = 10000000;  // The maximum allowed value for a + b + c
    const long long k_max = (max_sum - 2) / 6.5;  // Compute the maximum value for k based on max_sum
    const int num_threads = omp_get_max_threads();  // Get the number of available threads for parallel processing

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    long long total_triplets = 0;

    // Parallel loop using OpenMP to divide the work across available threads
    #pragma omp parallel for reduction(+:total_triplets) schedule(dynamic)
    for (long long k = 0; k <= k_max; ++k) {
        total_triplets += count_triplets(k, k, max_sum);
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the result
    std::cout << "Total triplets: " << total_triplets << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}