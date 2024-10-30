#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cmath>
#include <algorithm>

std::atomic<int64_t> triangle_count(0); // Use atomic for thread-safe increment
std::mutex print_mutex; // Mutex for printing

// Function to compute gcd
int64_t gcd(int64_t a, int64_t b) {
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

// Function to check for primitive triangles
void count_triangles(int64_t max_perimeter, int64_t start, int64_t end) {
    for (int64_t p = start; p < end; ++p) {
        for (int64_t a = 1; a <= p / 3; ++a) {
            for (int64_t b = a; b <= (p - a) / 2; ++b) {
                int64_t c = p - a - b;
                if (c >= b && a + b > c) { // Check triangle inequality
                    if (gcd(gcd(a, b), c) == 1) { // Check if primitive
                        int64_t count = triangle_count.fetch_add(1); // Atomic increment
                        if ((count + 1) % 1000000 == 0) { // Print every millionth triangle
                            std::lock_guard<std::mutex> guard(print_mutex); // Lock for safe printing
                            std::cout << "Found triangle #" << (count + 1) << ": (" << a << ", " << b << ", " << c << ")\n";
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int64_t max_perimeter = 10000000;
    const int num_threads = std::thread::hardware_concurrency(); // Get number of threads
    std::vector<std::thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads with dynamic load balancing
    for (int i = 0; i < num_threads; ++i) {
        int64_t range_start = (max_perimeter / num_threads) * i + 1;
        int64_t range_end = (max_perimeter / num_threads) * (i + 1) + 1;
        if (i == num_threads - 1) {
            range_end = max_perimeter + 1; // Ensure the last thread covers all remaining
        }
        threads.emplace_back(count_triangles, max_perimeter, range_start, range_end);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    // Output the total count of primitive triangles found
    std::cout << "Total primitive triangles found: " << triangle_count.load() << "\n";
    std::cout << "Execution time: " << duration.count() << " seconds.\n";

    return 0;
}