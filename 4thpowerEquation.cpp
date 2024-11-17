#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <algorithm>

constexpr int MIN_E = 1;
constexpr int MAX_E = 10000;
constexpr int MAX_N = 10000; // Daha düşük bir değer seçildi

typedef __int128 int128_t;

// Structure to hold solutions
struct Solution {
    int a, b, c, d, e;
};

// Mutex for synchronized console output
std::mutex cout_mutex;

// Atomic variable for dynamic work distribution
std::atomic<int> current_e(MIN_E);

// Precomputed fifth powers
std::vector<int128_t> fifth_powers;

// Precomputed sums of a^5 + b^5 and c^5 + d^5
std::vector<std::pair<int128_t, std::pair<int, int>>> ab_sums;
std::vector<std::pair<int128_t, std::pair<int, int>>> cd_sums;

// Function to compute a^5
inline int128_t compute_a5(int a) {
    return fifth_powers[a];
}

// Multithreaded precomputation of fifth powers
void precompute_fifth_powers(int thread_id, int num_threads) {
    int total = MAX_E + 1;
    int chunk_size = (total + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size;
    int end = std::min(start + chunk_size, total);

    for (int i = start; i < end; ++i) {
        int128_t i_int = static_cast<int128_t>(i);
        fifth_powers[i] = i_int * i_int * i_int * i_int * i_int;
    }
}

// Multithreaded precomputation of sums
void precompute_sums(int thread_id, int num_threads,
                     std::vector<std::pair<int128_t, std::pair<int, int>>>& local_ab_sums,
                     std::vector<std::pair<int128_t, std::pair<int, int>>>& local_cd_sums) {
    int total = MAX_N;
    int chunk_size = (total + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size + 1;
    int end = std::min(start + chunk_size, total + 1);

    // Precompute sums of a^5 + b^5
    for (int a = start; a < end; ++a) {
        int128_t a5 = compute_a5(a);
        for (int b = a; b <= MAX_N; ++b) { // Ensure a <= b
            int128_t b5 = compute_a5(b);
            int128_t sum_ab = a5 + b5;
            local_ab_sums.emplace_back(sum_ab, std::make_pair(a, b));
        }
    }

    // Precompute sums of c^5 + d^5
    for (int c = start; c < end; ++c) {
        int128_t c5 = compute_a5(c);
        for (int d = c; d <= MAX_N; ++d) { // Ensure c <= d
            int128_t d5 = compute_a5(d);
            int128_t sum_cd = c5 + d5;
            local_cd_sums.emplace_back(sum_cd, std::make_pair(c, d));
        }
    }
}

// Worker function for each thread to find and print solutions
void worker(int thread_id) {
    while (true) {
        int e = current_e.fetch_add(1);
        if (e > MAX_E) break;

        int128_t e5 = compute_a5(e);

        // Binary search for sum_cd
        for (const auto& ab_entry : ab_sums) {
            int128_t sum_ab = ab_entry.first;
            int128_t sum_cd = e5 - sum_ab;

            // Binary search in sorted cd_sums
            // Assuming cd_sums is sorted
            auto range = std::equal_range(cd_sums.begin(), cd_sums.end(), std::make_pair(sum_cd, std::make_pair(0,0)),
                [](const std::pair<int128_t, std::pair<int, int>>& a,
                   const std::pair<int128_t, std::pair<int, int>>& b) -> bool {
                    return a.first < b.first;
                });

            for (auto it = range.first; it != range.second; ++it) {
                int a = ab_entry.second.first;
                int b = ab_entry.second.second;
                int c = it->second.first;
                int d = it->second.second;

                // Ensure constraints a <= b <= c <= d <= e
                if (a <= b && b <= c && c <= d && d <= e) {
                    // Print the solution with thread synchronization
                    std::lock_guard<std::mutex> guard(cout_mutex);
                    std::cout << "Solution found: (a, b, c, d, e) = ("
                              << a << ", " << b << ", " << c << ", "
                              << d << ", " << e << ")\n";
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

    // Initialize fifth_powers vector
    fifth_powers.resize(MAX_E + 1);

    // Multithreaded precomputation of fifth powers
    std::cout << "Precomputing fifth powers up to " << MAX_E << " using " << n_threads << " threads..." << std::endl;
    {
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < n_threads; ++i) {
            threads.emplace_back(precompute_fifth_powers, i, n_threads);
        }
        for (auto& th : threads) {
            th.join();
        }
    }

    // Multithreaded precomputation of sums
    std::cout << "Precomputing sums of a^5 + b^5 and c^5 + d^5 up to " << MAX_N << " using " << n_threads << " threads..." << std::endl;
    std::vector<std::vector<std::pair<int128_t, std::pair<int, int>>>> local_ab_maps(n_threads);
    std::vector<std::vector<std::pair<int128_t, std::pair<int, int>>>> local_cd_maps(n_threads);

    {
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < n_threads; ++i) {
            threads.emplace_back(precompute_sums, i, n_threads, std::ref(local_ab_maps[i]), std::ref(local_cd_maps[i]));
        }
        for (auto& th : threads) {
            th.join();
        }
    }

    // Merge local maps into global vectors
    std::cout << "Merging local sums into global vectors..." << std::endl;
    for (unsigned int i = 0; i < n_threads; ++i) {
        ab_sums.insert(ab_sums.end(), local_ab_maps[i].begin(), local_ab_maps[i].end());
        cd_sums.insert(cd_sums.end(), local_cd_maps[i].begin(), local_cd_maps[i].end());
    }

    // Sort the cd_sums vector for binary search
    std::cout << "Sorting cd_sums for binary search..." << std::endl;
    std::sort(cd_sums.begin(), cd_sums.end(), 
        [](const std::pair<int128_t, std::pair<int, int>>& a,
           const std::pair<int128_t, std::pair<int, int>>& b) -> bool {
               return a.first < b.first;
        });

    // Launch worker threads to find solutions
    std::cout << "Launching worker threads to find solutions..." << std::endl;
    std::vector<std::thread> worker_threads;
    for (unsigned int i = 0; i < n_threads; ++i) {
        worker_threads.emplace_back(worker, i);
    }

    // Join worker threads
    for (auto& th : worker_threads) {
        th.join();
    }

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // Report total computation time
    std::cout << "All threads have completed their execution." << std::endl;
    std::cout << "Total computation time: " << diff.count() << " seconds." << std::endl;

    return 0;
}