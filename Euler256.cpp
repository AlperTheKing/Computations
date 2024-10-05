#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>

const int TARGET_T = 200;

// Mutex for thread-safe output
std::mutex cout_mutex;

// Function to compute T(s) for a given s
void compute_T_s(int start_s, int step, int& result_s) {
    for (int s = start_s; ; s += step) {
        int count = 0;

        // Skip odd s
        if (s % 2 != 0) continue;

        // Find all pairs (a, b) such that a * b = s and a <= b
        int sqrt_s = static_cast<int>(std::sqrt(s));
        for (int a = 1; a <= sqrt_s; ++a) {
            if (s % a != 0) continue;
            int b = s / a;
            if (a > b) continue; // Ensure a <= b

            // Check tatami-free condition
            // Tatami-free rooms occur when both a and b are even but not multiples of 4,
            // or one dimension is congruent to 2 mod 4 and the other is even
            bool condition = false;

            if (a % 2 == 0 && b % 2 == 0) {
                if ((a % 4 != 0 && b % 4 != 0) || (a % 4 == 2 || b % 4 == 2)) {
                    condition = true;
                }
            }

            if (condition) {
                count++;
            }
        }

        if (count == TARGET_T) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            result_s = s;
            break;
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    int result_s = 0;
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Start threads with different starting points to avoid overlap
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(compute_T_s, 2 + i * 2, num_threads * 2, std::ref(result_s));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    if (result_s != 0) {
        std::cout << "The smallest room size s for which T(s) = " << TARGET_T << " is: " << result_s << std::endl;
    } else {
        std::cout << "No result found." << std::endl;
    }

    std::cout << "Total execution time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}