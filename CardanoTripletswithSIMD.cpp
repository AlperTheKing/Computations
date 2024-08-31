#include <iostream>
#include <vector>
#include <tuple>
#include <omp.h>
#include <chrono>
#include <arm_neon.h>  // For NEON intrinsics

// Function to check the triplet condition using SIMD for b and c values
std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> process_range(unsigned long long a, unsigned long long r) {
    std::vector<std::tuple<unsigned long long, unsigned long long, unsigned long long>> results;
    
    uint64x2_t a_vec = vdupq_n_u64(a);
    uint64x2_t a2_vec = vmulq_u64(a_vec, a_vec);
    uint64x2_t a3_vec = vmulq_u64(a2_vec, a_vec);
    
    uint64x2_t term1 = vmulq_n_u64(a3_vec, 8);
    uint64x2_t term2 = vmulq_n_u64(a2_vec, 15);
    uint64x2_t term3 = vmulq_n_u64(a_vec, 6);

    for (unsigned long long b = 0; b < r; ++b) {
        uint64x2_t b_vec = vdupq_n_u64(b);
        uint64x2_t b2_vec = vmulq_u64(b_vec, b_vec);
        
        for (unsigned long long c = 0; c < r; c += 2) { // Process 2 c values in parallel
            uint64x2_t c_vec = {c, c+1};
            uint64x2_t bc_vec = vmulq_u64(b2_vec, c_vec);
            
            // Calculate the final term
            uint64x2_t term = vaddq_u64(vaddq_u64(term1, term2), term3);
            term = vsubq_u64(term, vmulq_n_u64(bc_vec, 27));

            // Check if the condition is met for either of the SIMD lanes
            if (vgetq_lane_u64(term, 0) == 1) {
                results.emplace_back(a, b, c);
            }
            if (vgetq_lane_u64(term, 1) == 1) {
                results.emplace_back(a, b, c + 1);
            }
        }
    }
    return results;
}

int main() {
    const unsigned long long r = 1000000; // Example range, adjust as needed
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