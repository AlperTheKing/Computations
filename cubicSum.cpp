#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <sstream>

// Define constants
constexpr int MIN_E = 1;
constexpr int MAX_E = 1000000;        // d < 1,000,000
constexpr int REPORT_INTERVAL = 10000; // Report progress every 10,000 d values

typedef __int128 int128_t;            // GCC extension for 128-bit signed integers

// Structure to hold solutions
struct Solution {
    int a, b, c, d;
};

// Mutex for synchronized console output
std::mutex cout_mutex;

// Precomputed fourth powers
std::vector<int128_t> fourth_powers;

// Function to compute a^4 (accesses precomputed fourth_powers)
inline int128_t compute_a4(int a) {
    return fourth_powers[a];
}

// Function to compute integer fourth root using binary search
inline int integer_fourth_root(int128_t value) {
    if (value < 0) return -1; // Negative numbers not handled
    int low = 0;
    int high = MAX_E;
    int mid;
    while (low <= high) {
        mid = low + (high - low) / 2;
        int128_t mid4 = static_cast<int128_t>(mid) * mid * mid * mid;
        if (mid4 == value) {
            return mid;
        } else if (mid4 < value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1; // Not found
}

// Function to precompute fourth powers sequentially
void precompute_fourth_powers_sequential() {
    for (int i = 0; i <= MAX_E; ++i) {
        int128_t i_int = static_cast<int128_t>(i);
        fourth_powers[i] = i_int * i_int * i_int * i_int;
    }
}

// Worker function for each thread to find and collect solutions
void worker(int thread_id, int start_d, int end_d, std::vector<Solution>& local_solutions) {
    for (int d = start_d; d < end_d; ++d) {
        int128_t d4 = compute_a4(d);

        for (int a = 1; a <= d; ++a) {
            int128_t a4 = compute_a4(a);
            if (a4 > d4) break; // Early termination

            for (int b = a; b <= d; ++b) {
                int128_t b4 = compute_a4(b);
                int128_t ab4 = a4 + b4;
                if (ab4 > d4) break; // Early termination

                int128_t c4 = d4 - ab4;
                if (c4 < b4) continue; // Since c >= b

                int c = integer_fourth_root(c4);
                if (c == -1) continue; // No integer fourth root
                if (c < b || c > d) continue; // Ensure c >= b and c <= d

                // Verify the equation
                int128_t total = a4 + static_cast<int128_t>(b) * b * b * b + static_cast<int128_t>(c) * c * c * c;
                if (total == d4) {
                    // Found a valid solution
                    Solution sol = {a, b, c, d};
                    local_solutions.push_back(sol);
                }
            }
        }
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine number of threads
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 8; // Default to 8 threads if unable to detect

    // Initialize fourth_powers vector with reserve and zero initialization
    fourth_powers.resize(MAX_E + 1, 0);

    // Sequential precomputation of fourth powers
    std::cout << "Precomputing fourth powers up to " << MAX_E << "..." << std::endl;
    precompute_fourth_powers_sequential();

    // Launch worker threads to find solutions
    std::cout << "Launching " << n_threads << " worker threads to find solutions..." << std::endl;
    std::vector<std::thread> worker_threads;
    std::vector<std::vector<Solution>> thread_solutions(n_threads); // Each thread has its own solution buffer

    // Determine the range of d values for each thread
    std::vector<std::pair<int, int>> thread_ranges;
    int range_size = (MAX_E - MIN_E + 1) / n_threads;
    int remainder = (MAX_E - MIN_E + 1) % n_threads;
    int current_start = MIN_E;

    for (unsigned int i = 0; i < n_threads; ++i) {
        int current_end = current_start + range_size;
        if (i < remainder) {
            current_end += 1;
        }
        thread_ranges.emplace_back(std::make_pair(current_start, current_end));
        current_start = current_end;
    }

    // Launch threads
    for (unsigned int i = 0; i < n_threads; ++i) {
        worker_threads.emplace_back(worker, i, thread_ranges[i].first, thread_ranges[i].second, std::ref(thread_solutions[i]));
    }

    // Progress reporting thread
    std::atomic<bool> done(false);
    std::thread progress_thread([&]() {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // Optional: Implement periodic progress reporting here
            // For simplicity, omitted in this version
        }
    });

    // Join worker threads
    for (auto& th : worker_threads) {
        th.join();
    }

    // Signal the progress thread to finish
    done.store(true);
    progress_thread.join();

    // Collect and print all solutions
    std::cout << "Collecting and printing solutions..." << std::endl;
    {
        std::lock_guard<std::mutex> guard(cout_mutex);
        for (unsigned int i = 0; i < n_threads; ++i) {
            for (const auto& sol : thread_solutions[i]) {
                std::cout << "Solution found: (a, b, c, d) = ("
                          << sol.a << ", " << sol.b << ", " << sol.c << ", " << sol.d << ")\n";
            }
        }
    }

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // Report total computation time
    std::cout << "All threads have completed their execution." << std::endl;
    std::cout << "Total computation time: " << diff.count() << " seconds." << std::endl;

    return 0;
}