#include <iostream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <limits>

const int TARGET_T = 200;        // Adjust this value as needed
const int MAX_DIMENSION = 50000; // Adjust based on computational resources

std::unordered_map<int64_t, int> T_s_count;
std::mutex t_s_mutex;

void compute_t_s(int thread_id, int num_threads) {
    for (int64_t m = 2 + thread_id; m <= MAX_DIMENSION; m += num_threads) {
        for (int64_t n = m; n <= MAX_DIMENSION; ++n) {
            bool valid = false;

            // Both dimensions are odd
            if (m % 2 == 1 && n % 2 == 1) {
                valid = true;
            }
            // Both dimensions are even, neither divisible by 4
            else if (m % 2 == 0 && n % 2 == 0 && m % 4 != 0 && n % 4 != 0) {
                valid = true;
            }
            // One dimension is odd, the other is even (even dimension not divisible by 4)
            else if (((m % 2 == 1 && n % 2 == 0) || (m % 2 == 0 && n % 2 == 1)) &&
                     ((m % 4 != 0 && m % 2 == 0) || (n % 4 != 0 && n % 2 == 0))) {
                valid = true;
            }

            if (valid) {
                int64_t s = m * n;

                // Update T_s_count in a thread-safe manner
                {
                    std::lock_guard<std::mutex> lock(t_s_mutex);
                    T_s_count[s]++;
                }
            }
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Start threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(compute_t_s, i, num_threads);
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // Find the maximum T(s) and the smallest s for which T(s) == TARGET_T
    int max_T_s = 0;
    int64_t s_at_max_T_s = 0;
    bool found = false;
    int64_t min_s = std::numeric_limits<int64_t>::max();
    for (const auto& kv : T_s_count) {
        if (kv.second > max_T_s) {
            max_T_s = kv.second;
            s_at_max_T_s = kv.first;
        }
        if (kv.second == TARGET_T) {
            if (kv.first < min_s) {
                min_s = kv.first;
                found = true;
            }
        }
    }

    std::cout << "Maximum T(s) found: " << max_T_s << " at s = " << s_at_max_T_s << std::endl;

    if (found) {
        std::cout << "The smallest room size s for which T(s) = " << TARGET_T << " is: " << min_s << std::endl;
    } else {
        std::cout << "No result found within the maximum dimension of " << MAX_DIMENSION << "." << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total execution time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}