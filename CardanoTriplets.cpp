#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>

int main() {
    long long max_sum = 1000000; // Initial maximum value for a + b + c
    long long max_k = max_sum / 6.5; // Maximum value of k based on the max_sum

    long long total_triplets = 0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (long long k = 0; k <= max_k; ++k) {
        long long a = 2 + 3 * k;
        long long k1 = k + 1;
        long long term = k1 * k1 * (8 * k + 5);

        for (long long b = 1; b * b <= term; ++b) {
            if (term % (b * b) == 0) {
                long long c = term / (b * b);
                long long abc_sum = a + b + c;

                // Ensure a, b, c are positive integers and their sum is within the limit
                if (a > 0 && b > 0 && c > 0 && abc_sum <= max_sum) {
                    #pragma omp atomic
                    total_triplets++;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the total number of triplets and the elapsed time
    std::cout << "Total triplets found: " << total_triplets << "\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}