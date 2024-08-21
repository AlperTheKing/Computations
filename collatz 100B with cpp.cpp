#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>

// Collatz steps calculation
int collatz_steps(long long n) {
    int steps = 0;
    while (n != 1) {
        if (n & 1) {  // n is odd
            n = 3 * n + 1;
        } else {
            n = n >> 1;  // n / 2
        }
        steps++;
    }
    return steps;
}

// Split the range into subranges for each thread
std::vector<std::pair<long long, long long>> split_range(long long start, long long end, int num_splits) {
    long long step = (end - start) / num_splits;
    std::vector<std::pair<long long, long long>> ranges;
    ranges.reserve(num_splits); // Reserve memory to avoid reallocations
    
    for (int i = 0; i < num_splits; ++i) {
        long long range_start = start + i * step;
        long long range_end = (i == num_splits - 1) ? end : range_start + step;
        ranges.emplace_back(range_start, range_end);
    }
    
    return ranges;
}

int main() {
    std::vector<std::pair<long long, long long>> groups = {
        {1, 10},
        {10, 100},
        {100, 1000},
        {1000, 10000},
        {10000, 100000},
        {100000, 1000000},
        {1000000, 10000000},
        {10000000, 100000000},
        {100000000, 1000000000},
        {1000000000, 10000000000},
        {10000000000, 100000000000}
    };

    const int num_threads = std::thread::hardware_concurrency();

    for (const auto& group : groups) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto ranges = split_range(group.first, group.second, num_threads);
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        std::vector<int> max_steps(num_threads);
        std::vector<long long> number_with_max_steps(num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                max_steps[i] = 0;
                number_with_max_steps[i] = 0;
                for (long long j = ranges[i].first; j < ranges[i].second; ++j) {
                    int steps = collatz_steps(j);
                    if (steps > max_steps[i]) {
                        max_steps[i] = steps;
                        number_with_max_steps[i] = j;
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }

        long long max_number = 0;
        int max_step_count = 0;

        for (int i = 0; i < num_threads; ++i) {
            if (max_steps[i] > max_step_count) {
                max_step_count = max_steps[i];
                max_number = number_with_max_steps[i];
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Range " << group.first << " to " << group.second << ":\n";
        std::cout << "  Number with max steps: " << max_number << " (" << max_step_count << " steps)\n";
        std::cout << "Computation time: " << elapsed.count() << " seconds\n\n";
    }

    return 0;
}