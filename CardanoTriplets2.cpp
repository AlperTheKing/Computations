#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <tuple>
#include <omp.h>

int main() {
    const long long max_sum = 10000000;  // The maximum allowed value for a + b + c
    const long long k_max = (max_sum - 2) / 6.5;  // Compute the maximum value for k based on max_sum
    long long total_triplets = 0;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        long long local_triplets = 0;  // Local count for each thread

        #pragma omp for reduction(+:total_triplets) schedule(dynamic)
        for (long long k = 0; k <= k_max; ++k) {
            long long a = 3 * k + 2;  // a = 3k + 2
            long long term = (k + 1) * (k + 1) * (8 * k + 5);  // b^2 * c = (k + 1)^2 * (8k + 5)

            for (long long b = 1; b * b <= term; ++b) {
                if (term % (b * b) == 0) {  // Check if b^2 divides term
                    long long c = term / (b * b);  // Calculate c

                    // Check if the sum a + b + c is within the allowed limit
                    if (a + b + c <= max_sum) {
                        local_triplets++;
                    }
                }
            }
        }

        #pragma omp critical
        total_triplets += local_triplets;
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the result
    std::cout << "Total triplets: " << total_triplets << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}