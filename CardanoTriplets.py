#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <vector>

long long count_triplets(long long start_k, long long end_k, long long max_sum) {
    long long total_count = 0;

    for (long long k = start_k; k <= end_k; ++k) {
        long long a = 3 * k + 2;  // a = 3k + 2
        long long term = (k + 1) * (k + 1) * (8 * k + 5);  // b^2 * c = (k + 1)^2 * (8k + 5)

        for (long long b = 1; b * b <= term; ++b) {
            if (term % (b * b) == 0) {  // Check if b^2 divides term
                long long c = term / (b * b);
                if (a + b + c <= max_sum) {
                    total_count++;
                }
            }
        }
    }

    return total_count;
}

int main() {
    long long max_sum = 1000;  // Modify this value as needed
    long long k_max = (max_sum - 2) / 3;  // Calculate maximum k value
    int num_threads = omp_get_max_threads();  // Get the number of available threads

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    long long total_triplets = 0;

    #pragma omp parallel for reduction(+:total_triplets)
    for (int i = 0; i < num_threads; ++i) {
        long long start_k = i;
        long long end_k = k_max;
        total_triplets += count_triplets(start_k, end_k, max_sum);
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the result
    std::cout << "Total triplets: " << total_triplets << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}