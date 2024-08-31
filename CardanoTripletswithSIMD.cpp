#include <iostream>
#include <vector>
#include <tuple>
#include <omp.h>
#include <chrono>
#include <algorithm>

// Function to check the triplet condition using scalar operations
std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> process_range(unsigned long long a, unsigned long long r) {
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> results;

    // Precompute parts of the term for efficiency
    unsigned long long a3 = 8 * a * a * a;
    unsigned long long a2 = 15 * a * a;
    unsigned long long a1 = 6 * a;

    for (unsigned long long b = 0; b < r; ++b) {
        unsigned long long b2 = b * b;
        for (unsigned long long c = 0; c < r; ++c) {
            unsigned long long term = a3 + a2 + a1 - 27 * b2 * c;
            if (term == 1) {
                results.emplace_back(a, b, c);
            }
        }
    }
    return results;
}

int main() {
    const unsigned long long r = 10000; // Example range, adjust as needed
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> found_triplets;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop using OpenMP
    #pragma omp parallel
    {
        std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> local_results;

        #pragma omp for schedule(dynamic) nowait
        for (unsigned long long a = 0; a < r; ++a) {
            auto partial_results = process_range(a, r);
            local_results.insert(local_results.end(), partial_results.begin(), partial_results.end());
        }

        // Merge the results from each thread into the main vector
        #pragma omp critical
        found_triplets.insert(found_triplets.end(), local_results.begin(), local_results.end());
    }

    // Sort triplets by the first element (a) in ascending order
    std::sort(found_triplets.begin(), found_triplets.end());

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print found triplets
    for (const auto& triplet : found_triplets) {
        std::cout << std::get<0>(triplet) << ", " << std::get<1>(triplet) << ", " << std::get<2>(triplet) << std::endl;
    }

    // Print the number of triplets found
    std::cout << found_triplets.size() << " triplet(s) found" << std::endl;

    // Print the elapsed time
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}