#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_set>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>

// Helper function to convert __int128 to string
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

// Newton's method to compute square root of S using only integers
__int128 newton_sqrt(__int128 S) {
    if (S == 0) return 0;
    __int128 x = S;
    __int128 precision = 1;  // Precision for integers
    while (true) {
        __int128 next_x = (x + S / x) / 2;
        if (abs(next_x - x) < precision) return next_x;
        x = next_x;
    }
}

// Precomputed primes of the form 4k+1 and their a^2 + b^2 representations (a <= b enforced)
const std::vector<int> primes = {5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149};
const std::vector<std::pair<int, int>> prime_factors = {
    {1, 2}, {2, 3}, {1, 4}, {2, 5}, {1, 6}, {4, 5}, {2, 7}, {5, 6}, {3, 8}, {5, 8}, {4, 9},
    {1, 10}, {6, 7}, {7, 8}, {2, 11}, {5, 12}
};

// Use composition law to combine two sums of squares in multiple ways
std::vector<std::pair<int, int>> combine_squares(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
    std::vector<std::pair<int, int>> combinations;
    int a1 = p1.first, b1 = p1.second;
    int a2 = p2.first, b2 = p2.second;

    // All combinations of signs for (a1*a2 - b1*b2, a1*b2 + a2*b1)
    combinations.emplace_back(std::abs(a1 * a2 - b1 * b2), std::abs(a1 * b2 + a2 * b1));
    combinations.emplace_back(std::abs(a1 * a2 + b1 * b2), std::abs(a1 * b2 - a2 * b1));

    // Ensure a <= b
    for (auto& comb : combinations) {
        if (comb.first > comb.second) std::swap(comb.first, comb.second);
    }

    return combinations;
}

// Validate a^2 + b^2 = N using Newton's method for square root
bool validate_pair(int a, __int128 N) {
    // Calculate N - a^2
    __int128 N_minus_a2 = N - __int128(a) * a;
    if (N_minus_a2 < 0) return false;  // If N - a^2 < 0, no valid b exists
    
    // Calculate b using Newton's method
    __int128 b = newton_sqrt(N_minus_a2);
    
    // Check if b is an integer and satisfies b^2 + a^2 = N
    return (b * b == N_minus_a2);
}

// Generate all square-free N values and store results in a map sorted by N
void generate_square_free_N(std::map<__int128, std::vector<std::pair<int, int>>>& result_map) {
    size_t num_primes = primes.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 1; i < (1 << num_primes); ++i) {
        __int128 N = 1;
        std::vector<std::vector<std::pair<int, int>>> factors;
        
        // For each subset of primes, calculate the product and factorize using precomputed values
        for (size_t j = 0; j < num_primes; ++j) {
            if (i & (1 << j)) {
                N *= primes[j];
                factors.push_back({prime_factors[j]});  // Use precomputed (a, b) pairs
            }
        }

        // Now combine all factors using the composition law in multiple ways
        std::vector<std::pair<int, int>> current_combinations = factors[0];
        for (size_t k = 1; k < factors.size(); ++k) {
            std::vector<std::pair<int, int>> new_combinations;
            for (const auto& comb1 : current_combinations) {
                for (const auto& comb2 : factors[k]) {
                    auto combined = combine_squares(comb1, comb2);
                    new_combinations.insert(new_combinations.end(), combined.begin(), combined.end());
                }
            }
            current_combinations = new_combinations;
        }

        #pragma omp critical
        {
            // Store all unique solutions in a map sorted by N
            std::unordered_set<std::string> seen_pairs;
            for (const auto& comb : current_combinations) {
                if (validate_pair(comb.first, N)) {  // Validate (a, b) pair using new approach
                    std::string pair_str = std::to_string(std::min(comb.first, comb.second)) + "," + std::to_string(std::max(comb.first, comb.second));
                    if (seen_pairs.find(pair_str) == seen_pairs.end()) {
                        result_map[N].push_back({comb.first, comb.second});
                        seen_pairs.insert(pair_str); // Mark the pair as seen
                    }
                }
            }
        }
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Map to store results sorted by N (std::map keeps keys in sorted order)
    std::map<__int128, std::vector<std::pair<int, int>>> result_map;

    // Generate all square-free N values and store in the map
    generate_square_free_N(result_map);

    // Output file
    std::ofstream file("SumofSquares_Sorted.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file SumofSquares_Sorted.txt for writing." << std::endl;
        return 1;
    }

    // Variable to store the total sum of a's
    __int128 total_sum_of_a = 0;

    // Write results sorted by N to the file
    for (const auto& entry : result_map) {
        file << "N = " << int128_to_string(entry.first) << " : ";
        for (const auto& pair : entry.second) {
            file << "(" << pair.first << "," << pair.second << ") ";
            total_sum_of_a += pair.first;  // Accumulate the sum of a's
        }
        file << std::endl;
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Write the total sum of a's and the elapsed time to the file
    file << "Total sum of a values: " << int128_to_string(total_sum_of_a) << std::endl;
    file << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // Close the file
    file.close();

    // Output results
    std::cout << "Total sum of a values: " << int128_to_string(total_sum_of_a) << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}