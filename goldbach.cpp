// goldbach.cpp

#include <iostream>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdint>

using namespace std;

// Corrected: Use '__uint128_t' and define 'MAX_K' without exceeding 64-bit limits
const __uint128_t MAX_K = (__uint128_t(1) * 1000000000000ULL * 1000000ULL * 1000ULL); // 1e21
const uint64_t MAX_M = static_cast<uint64_t>(sqrt(static_cast<double>(MAX_K))) + 1;

// Removed unused mutex_sum
mutex mutex_k_set;

unordered_set<uint64_t> global_k_set;
long double global_sum = 0.0;

// Function to check if a number is a perfect power
bool is_perfect_power(uint64_t m) {
    if (m <= 1) return false;
    for (uint64_t e = 2; e <= log2(m); ++e) {
        double root_d = pow(static_cast<double>(m), 1.0 / e);
        uint64_t root = round(root_d);
        __uint128_t power = 1;
        for (uint64_t i = 0; i < e; ++i) {
            power *= root;
            if (power > m) break; // Early exit if power exceeds m
        }
        if (power == m) {
            return true;
        }
    }
    return false;
}

// Worker function for multithreading
void worker(uint64_t start_m, uint64_t end_m) {
    unordered_set<uint64_t> local_k_set;

    for (uint64_t m = start_m; m <= end_m; ++m) {
        if (is_perfect_power(m)) {
            continue;
        }
        __uint128_t k = m;
        for (uint64_t n = 2; ; ++n) {
            k *= m;
            if (k > MAX_K) {
                break;
            }
            uint64_t k_uint64 = static_cast<uint64_t>(k);
            local_k_set.insert(k_uint64);
        }
    }

    // Lock and update the global set and sum
    lock_guard<mutex> lock(mutex_k_set);
    for (const auto& k : local_k_set) {
        if (global_k_set.insert(k).second) {
            global_sum += 1.0L / (k - 1);
        }
    }
}

int main() {
    auto start_time = chrono::high_resolution_clock::now();

    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    vector<thread> threads;
    uint64_t range = MAX_M / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        uint64_t start_m = 2 + i * range;
        uint64_t end_m = (i == num_threads - 1) ? MAX_M : start_m + range - 1;
        threads.emplace_back(worker, start_m, end_m);
    }

    for (auto& th : threads) {
        th.join();
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout.precision(15);
    cout << "Sum S = " << global_sum << endl;
    cout << "Total unique k values: " << global_k_set.size() << endl;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    return 0;
}