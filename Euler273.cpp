#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <chrono>

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

// Function to find sum of two squares for a prime of the form 4k + 1
std::vector<std::pair<int, int>> sum_of_two_squares(int p) {
    std::vector<std::pair<int, int>> results;
    for (int a = 1; a <= std::sqrt(p); ++a) {
        int b_squared = p - a * a;
        int b = std::sqrt(b_squared);
        if (b * b == b_squared) {
            results.emplace_back(a, b);  // Return all valid pairs (a, b)
        }
    }
    return results;
}

// Generate primes of the form 4k + 1 using sieve of Eratosthenes
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

// Use composition law to combine two sums of squares in multiple ways
std::vector<std::pair<int, int>> combine_squares(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
    std::vector<std::pair<int, int>> combinations;
    int a1 = p1.first, b1 = p1.second;
    int a2 = p2.first, b2 = p2.second;

    // All combinations of signs for (a1*a2 - b1*b2, a1*b2 + a2*b1)
    combinations.emplace_back(std::abs(a1 * a2 - b1 * b2), std::abs(a1 * b2 + a2 * b1));
    combinations.emplace_back(std::abs(a1 * a2 + b1 * b2), std::abs(a1 * b2 - a2 * b1));

    return combinations;
}

// Generate all square-free N values and write results to file
void generate_square_free_N(const std::vector<int>& primes, std::ofstream& file, __int128& total_sum_of_a) {
    size_t num_primes = primes.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 1; i < (1 << num_primes); ++i) {
        __int128 N = 1;
        std::vector<std::vector<std::pair<int, int>>> factors;
        
        // For each subset of primes, calculate the product and factorize using Gaussian integers
        for (size_t j = 0; j < num_primes; ++j) {
            if (i & (1 << j)) {
                N *= primes[j];
                factors.push_back(sum_of_two_squares(primes[j]));
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
            // Store all unique solutions directly to file to avoid memory overflow
            std::unordered_set<std::string> seen_pairs;
            for (const auto& comb : current_combinations) {
                std::string pair_str = std::to_string(std::min(comb.first, comb.second)) + "," + std::to_string(std::max(comb.first, comb.second));
                if (seen_pairs.find(pair_str) == seen_pairs.end()) {
                    file << "N = " << int128_to_string(N) << ", a = " << comb.first << ", b = " << comb.second << std::endl;
                    seen_pairs.insert(pair_str); // Mark the pair as seen
                    total_sum_of_a += comb.first;  // Accumulate the sum of a's
                }
            }
        }
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Limit for primes of the form 4k + 1
    int prime_limit = 150;

    // Generate primes of the form 4k + 1 using sieve
    std::vector<int> primes = sieve_of_eratosthenes(prime_limit);

    // Output file
    std::ofstream file("SumofSquares.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file SumofSquares.txt for writing." << std::endl;
        return 1;
    }

    // Variable to store the total sum of a's
    __int128 total_sum_of_a = 0;

    // Generate all square-free N values and write directly to file
    generate_square_free_N(primes, file, total_sum_of_a);

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