#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional> // For std::hash

#define MIN_N 62498
#define MAX_N 100000

typedef __int128 int128_t; // GCC extension for 128-bit integers

// Custom hash function for __int128
namespace std {
    template<>
    struct hash<__int128> {
        std::size_t operator()(const __int128& x) const noexcept {
            // Since size_t is typically 64 bits, we can split __int128 into two 64-bit parts
            uint64_t high = static_cast<uint64_t>(x >> 64);
            uint64_t low = static_cast<uint64_t>(x);
            // Combine the two parts using a hash combiner
            return std::hash<uint64_t>{}(high) ^ (std::hash<uint64_t>{}(low) << 1);
        }
    };
}

std::vector<int128_t> fourth_powers(MAX_N + 1);
std::unordered_map<int128_t, int> fourth_power_map;

std::mutex output_mutex;
std::atomic<int> current_e(MIN_N);

std::string int128_to_string(int128_t x) {
    if (x == 0) return "0";
    bool neg = x < 0;
    if (neg) x = -x;
    std::string s;
    while (x > 0) {
        s = char('0' + static_cast<int>(x % 10)) + s;
        x /= 10;
    }
    if (neg) s = "-" + s;
    return s;
}

void worker() {
    int e;
    while ((e = current_e.fetch_add(1)) <= MAX_N) {
        int128_t e4 = fourth_powers[e];
        for (int d = 1; d <= e; ++d) {
            int128_t d4 = fourth_powers[d];
            if (d4 > e4) break;
            for (int c = 1; c <= d; ++c) {
                int128_t c4 = fourth_powers[c];
                if (d4 + c4 > e4) break;
                int128_t s = e4 - d4 - c4;
                if (s < 0) continue;
                // Since a ≤ b ≤ c, and a, b ≤ c, we only need to check a ≤ b ≤ c
                for (int a = 1; a <= c; ++a) {
                    int128_t a4 = fourth_powers[a];
                    int128_t b4 = s - a4;
                    if (b4 < a4) continue; // Ensure a ≤ b
                    auto it = fourth_power_map.find(b4);
                    if (it != fourth_power_map.end()) {
                        int b = it->second;
                        if (b >= a && b <= c) {
                            // Output the solution
                            std::lock_guard<std::mutex> guard(output_mutex);
                            std::cout << "(" << a << ", " << b << ", " << c << ", " << d << ", " << e << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Precompute fourth powers
    for (int i = 1; i <= MAX_N; ++i) {
        int128_t i4 = static_cast<int128_t>(i) * i * i * i;
        fourth_powers[i] = i4;
        fourth_power_map[i4] = i;
    }

    // Measure start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine number of threads
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4; // Default to 4 threads if unable to detect

    std::vector<std::thread> threads;

    // Launch threads with dynamic load balancing
    for (unsigned int i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    // Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}