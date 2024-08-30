#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <omp.h>

int main() {
    int total_count = 0;
    std::vector<std::tuple<int, int, int>> all_triplets;
    double tolerance = 1e-9;

    // Parallel processing with OpenMP
    #pragma omp parallel for reduction(+:total_count)
    for (int a = 0; a < 1000; ++a) {
        std::vector<std::tuple<int, int, int>> local_triplets;
        for (int b = 0; b < 1000; ++b) {
            for (int c = 0; c < 1000; ++c) {
                double term1 = std::cbrt(a + b * std::sqrt(c));
                double term2 = std::cbrt(a - b * std::sqrt(c));

                if (std::abs((term1 + term2) - 1) < tolerance) {
                    local_triplets.push_back(std::make_tuple(a, b, c));
                    total_count++;
                }
            }
        }

        #pragma omp critical
        {
            all_triplets.insert(all_triplets.end(), local_triplets.begin(), local_triplets.end());
        }
    }

    // Print the triplets
    for (const auto& triplet : all_triplets) {
        std::cout << "Triplet: (" << std::get<0>(triplet) << ", " << std::get<1>(triplet) << ", " << std::get<2>(triplet) << ")\n";
    }

    std::cout << total_count << " triplet found" << std::endl;

    return 0;
}