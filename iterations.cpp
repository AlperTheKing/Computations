#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <mutex>

using namespace std;
using namespace std::chrono;

// Struct to store each solution
struct Solution {
    unsigned long long k;
    unsigned long long a;
    unsigned long long m;
    unsigned long long n;

    // Constructor to initialize the struct
    Solution(unsigned long long k, unsigned long long a, unsigned long long m, unsigned long long n)
        : k(k), a(a), m(m), n(n) {}
};

// Mutex for thread-safe access to shared vector
mutex mtx;

// Function to find solutions
void find_solutions(int k_start, int k_end, vector<Solution>& solutions, unsigned long long kmax) {
    for (unsigned long long k = k_start; k <= k_end; ++k) {
        for (unsigned long long a = 0; a <= k / 2; ++a) {
            unsigned long long max_m;

            // Conditional logic for determining max_m based on the value of a
            if (a == 0) {
                max_m = sqrt(kmax / 2);  // Use kmax when calculating m for a = 0
            } else {
                max_m = kmax / 1000;     // When a != 0, m goes up to k/1000
            }

            for (unsigned long long m = 1; m <= max_m; ++m) {
                unsigned long long left_side = (2 * k + a) * m * (a + m + 1);
                unsigned long long right_side = k * k;

                if (left_side == right_side) {
                    lock_guard<mutex> guard(mtx);  // Ensure thread-safe access
                    solutions.push_back(Solution(k, a, m, k + a));  // Store the solution
                }
            }
        }
    }
}

int main() {
    auto start = high_resolution_clock::now();

    vector<Solution> solutions;  // To store all the solutions

    // Automatically determine the number of cores from the system
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;  // Fallback in case the function returns 0

    unsigned long long kmax = 1000000;
    unsigned long long range_per_thread = kmax / num_threads;

    // Create threads for parallel execution
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        unsigned long long k_start = i * range_per_thread + 1;
        unsigned long long k_end = (i + 1) * range_per_thread;
        threads.push_back(thread(find_solutions, k_start, k_end, ref(solutions), kmax));
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    // Sort the solutions by k in ascending order
    sort(solutions.begin(), solutions.end(), [](const Solution& s1, const Solution& s2) {
        return s1.k < s2.k;
    });

    // Calculate the total sum and count of all k values
    unsigned long long total_k_sum = 0;
    int total_k_count = 0;

    for (const auto& sol : solutions) {
        total_k_sum += sol.k;
        total_k_count++;
    }

    // Print the total count and sum of all k values
    cout << "Total number of k values: " << total_k_count << endl;
    cout << "Total sum of all k values: " << total_k_sum << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);  // Use seconds for time measurement

    // Print the execution time in seconds
    cout << "Execution time: " << duration.count() << " seconds" << endl;

    return 0;
}