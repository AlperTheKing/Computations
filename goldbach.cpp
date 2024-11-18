// sum_perfect_powers.cpp

#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <boost/multiprecision/cpp_int.hpp>

// Use Boost multiprecision for large integers
using boost::multiprecision::uint128_t;

const uint64_t MAX_K = 1000000000000000000ULL; // 1e12
const uint64_t MAX_M = static_cast<uint64_t>(std::pow(MAX_K, 0.5)) + 1;

// Mutex for thread-safe access to the global set and sum
std::mutex mutex_k_set;
std::mutex mutex_sum;

// Global set to store unique k values
std::unordered_set<uint64_t> global_k_set;

// Global sum
long double global_sum = 0.0;

// Function to check if a number is a perfect power
bool is_perfect_power(uint64_t m) {
    uint64_t max_exponent = static_cast<uint64_t>(std::log2(m)) + 1;
    for (uint64_t exponent = 2; exponent <= max_exponent; ++exponent) {
        uint64_t low = 1;
        uint64_t high = m;
        while (low <= high) {
            uint64_t mid = low + (high - low) / 2;
            __uint128_t mid_e_power = 1;
            for (uint64_t i = 0; i < exponent; ++i) {
                mid_e_power *= mid;
                if (mid_e_power > m) {
                    break;
                }
            }
            if (mid_e_power == m) {
                return true;
            } else if (mid_e_power < m) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    return false;
}

// Worker function for each thread
void worker(uint64_t start_m, uint64_t end_m) {
    std::unordered_set<uint64_t> local_k_set;
    long double local_sum = 0.0;

    for (uint64_t m = start_m; m <= end_m; ++m) {
        if (is_perfect_power(m)) {
            continue;
        }
        uint128_t k = m * m; // Start with n = 2
        uint64_t n = 2;
        while (k <= MAX_K) {
            uint64_t k_uint64 = static_cast<uint64_t>(k);
            local_k_set.insert(k_uint64);
            n += 1;
            k *= m;
            if (k > MAX_K) {
                break;
            }
        }
    }

    // Lock and update the global set and sum
    {
        std::lock_guard<std::mutex> lock(mutex_k_set);
        for (const auto& k : local_k_set) {
            if (global_k_set.insert(k).second) {
                global_sum += 1.0L / (k - 1);
            }
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default to 4 threads if hardware_concurrency is not defined

    std::vector<std::thread> threads;
    uint64_t range = MAX_M / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        uint64_t start_m = 2 + i * range;
        uint64_t end_m = (i == num_threads - 1) ? MAX_M : start_m + range - 1;
        threads.emplace_back(worker, start_m, end_m);
    }

    for (auto& th : threads) {
        th.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout.precision(15);
    std::cout << "Sum S = " << global_sum << std::endl;
    std::cout << "Total unique k values: " << global_k_set.size() << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}