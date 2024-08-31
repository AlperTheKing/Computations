#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>
#include <chrono>

// Function to check the triplet condition
bool check_triplet(unsigned long long a, unsigned long long b, unsigned long long c) {
    unsigned long long term = (8 * std::pow(a, 3)) + (15 * std::pow(a, 2)) + (6 * a) - (27 * std::pow(b, 2) * c);
    return (term == 1);
}

// Function to process a range of values for a
std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> process_range(unsigned long long a, unsigned long long r) {
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> results;
    for (unsigned long long b = 0; b < r; ++b) {
        for (unsigned long long c = 0; c < r; ++c) {
            if (check_triplet(a, b, c)) {
                results.emplace_back(a, b, c);
            }
        }
    }
    return results;
}

int main() {
    const unsigned long long r = 1000000; // You can increase this to 110 million, but it will take significant time
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> found_triplets;

    // Zaman ölçümünü başlat
    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (unsigned long long a = 0; a < r; ++a) {
        std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> local_results = process_range(a, r);
        #pragma omp critical
        {
            found_triplets.insert(found_triplets.end(), local_results.begin(), local_results.end());
        }
    }

    // Zaman ölçümünü bitir
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Bulunan tripletleri ekrana yazdır
    for (const auto& triplet : found_triplets) {
        std::cout << std::get<0>(triplet) << ", " << std::get<1>(triplet) << ", " << std::get<2>(triplet) << std::endl;
    }

    // Triplet sayısını yazdır
    std::cout << found_triplets.size() << " triplet(s) found" << std::endl;

    // Toplam geçen süreyi en sona yazdır
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}