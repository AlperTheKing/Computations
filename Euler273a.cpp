#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <fstream>
#include <chrono>
#include <set>
#include <map>
#include <algorithm>

// Global mutex to protect shared resources in a multithreaded environment
std::mutex mtx;

// List of primes used for generating combinations of N
std::vector<int> primes = {5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149};

// Global map to store the results of (N, Gaussian integer pairs) and prime factorization information
std::map<__int128, std::set<std::pair<int, int>>> results;
std::map<__int128, std::string> prime_factors;

// Variable to store the total sum of all 'a' values across the Gaussian integer pairs
__int128 total_sum_of_a = 0;

// Function to find Gaussian integer pairs (a, b) for prime p, ensuring a <= b
std::vector<std::pair<int, int>> find_gaussian_pairs(int p) {
    std::vector<std::pair<int, int>> pairs;

    // Find (a, b) such that a^2 + b^2 = p, and ensure a <= b
    for (int a = 0; a * a <= p; ++a) {
        int b_squared = p - a * a;
        int b = static_cast<int>(sqrt(b_squared));
        if (b * b == b_squared && a <= b) {
            pairs.emplace_back(a, b);  // Ensure a <= b
        }
    }
    return pairs;
}

// Function to combine two sets of Gaussian integer pairs (a1, b1) and (a2, b2) for products of primes
std::set<std::pair<int, int>> combine_gaussian_pairs(const std::set<std::pair<int, int>>& pairs1,
                                                     const std::set<std::pair<int, int>>& pairs2) {
    std::set<std::pair<int, int>> result;
    for (const auto& [a1, b1] : pairs1) {
        for (const auto& [a2, b2] : pairs2) {
            // Consider both (a1*b2 + a2*b1) and (a1*b2 - a2*b1) combinations, ensuring a <= b
            int a_plus = std::abs(a1 * a2 - b1 * b2);
            int b_plus = std::abs(a1 * b2 + a2 * b1);
            if (a_plus <= b_plus) result.emplace(a_plus, b_plus);
            else result.emplace(b_plus, a_plus);  // Ensure a <= b

            // Also consider the reverse combination
            int a_minus = std::abs(a1 * a2 + b1 * b2);
            int b_minus = std::abs(a1 * b2 - a2 * b1);
            if (a_minus <= b_minus) result.emplace(a_minus, b_minus);
            else result.emplace(b_minus, a_minus);  // Ensure a <= b
        }
    }
    return result;
}

// Function to find Gaussian integer pairs (a, b) for N, and store the results
void find_solutions_gaussian(__int128 original_N, const std::string& factors) {
    std::set<std::pair<int, int>> solutions;
    __int128 N = original_N;  // Retain the original value of N

    for (int prime : primes) {
        if (N % prime == 0) {
            auto prime_pairs = find_gaussian_pairs(prime);
            std::set<std::pair<int, int>> prime_pair_set(prime_pairs.begin(), prime_pairs.end());
            N /= prime;

            if (solutions.empty()) {
                solutions = prime_pair_set;  // Initialize with the first prime factor's pairs
            } else {
                solutions = combine_gaussian_pairs(solutions, prime_pair_set);
            }
        }
    }

    // Add results to the global map if valid solutions are found
    if (!solutions.empty()) {
        __int128 sum_of_a = 0;
        for (const auto& [a, b] : solutions) {
            sum_of_a += a;
        }

        std::lock_guard<std::mutex> lock(mtx);
        results[original_N] = solutions;  // Store solutions with the original N value
        prime_factors[original_N] = factors;  // Store factorization
        total_sum_of_a += sum_of_a;
    }
}

// Custom function to print __int128 values (since std::cout does not support __int128 natively)
void print_int128(__int128 value) {
    if (value == 0) {
        std::cout << "0";
        return;
    }
    if (value < 0) {
        std::cout << "-";
        value = -value;
    }

    std::string result;
    while (value > 0) {
        result += '0' + (value % 10);
        value /= 10;
    }

    std::reverse(result.begin(), result.end());
    std::cout << result;
}

// Function to distribute work across threads for parallel processing
void worker_thread(int start, int end, const std::vector<__int128>& all_combinations, const std::vector<std::string>& all_factors) {
    for (int i = start; i < end; ++i) {
        find_solutions_gaussian(all_combinations[i], all_factors[i]);
    }
}

// Function to generate all squarefree N values and find Gaussian integer solutions
void generate_and_find_gaussian() {
    size_t num_primes = primes.size();
    std::vector<__int128> all_combinations;
    std::vector<std::string> all_factors;

    // Use bitmasking to generate all non-empty combinations of primes
    for (int mask = 1; mask < (1 << num_primes); ++mask) {
        __int128 N = 1;
        std::string factors;
        for (size_t i = 0; i < num_primes; ++i) {
            if (mask & (1 << i)) {
                N *= primes[i];
                factors += (factors.empty() ? "" : "*") + std::to_string(primes[i]);
            }
        }
        all_combinations.push_back(N);
        all_factors.push_back(factors);
    }

    // Get the number of available threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2;  // Default to 2 if hardware_concurrency is not supported
    }

    std::vector<std::thread> threads;
    int work_per_thread = all_combinations.size() / num_threads;
    int remaining_work = all_combinations.size() % num_threads;

    // Assign tasks to threads
    int start = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int end = start + work_per_thread + (i < remaining_work ? 1 : 0);  // Handle remaining work
        threads.emplace_back(worker_thread, start, end, std::ref(all_combinations), std::ref(all_factors));
        start = end;
    }

    // Join threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

int main() {
    // Measure overall execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Measure time for generating and finding Gaussian integer pairs
    auto computation_start = std::chrono::high_resolution_clock::now();
    generate_and_find_gaussian();
    auto computation_end = std::chrono::high_resolution_clock::now();

    // Write results to file (SumofSquares.txt)
    std::ofstream output_file("SumofSquares.txt");
    if (output_file.is_open()) {
        for (const auto& [N, solutions] : results) {
            output_file << "N = ";
            print_int128(N);  // Custom print for __int128
            output_file << " (" << prime_factors[N] << "): ";
            for (const auto& [a, b] : solutions) {
                output_file << "(" << a << "," << b << ") ";
            }
            output_file << std::endl;
        }
        output_file.close();
    }

    // Compute the elapsed time for computation
    std::chrono::duration<double> computation_duration = computation_end - computation_start;

    // Print the computation time
    std::cout << "Computation time: " << computation_duration.count() << " seconds" << std::endl;

    // Print the total sum of all 'a' values
    std::cout << "Total sum of all 'a' values: ";
    print_int128(total_sum_of_a);  // Custom print for __int128
    std::cout << std::endl;

    return 0;
}