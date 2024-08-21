#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <chrono>

int collatz_steps(long long n) {
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

std::pair<long long, int> calculate_collatz_range(long long start, long long end) {
    int local_max_steps = 0;
    long long local_number_with_max_steps = 0;
    
    for (long long i = start; i < end; ++i) {
        int steps = collatz_steps(i);
        if (steps > local_max_steps) {
            local_max_steps = steps;
            local_number_with_max_steps = i;
        }
    }
    
    return std::make_pair(local_number_with_max_steps, local_max_steps);
}

std::vector<std::pair<long long, long long>> split_range(long long start, long long end, int num_splits) {
    long long step = (end - start) / num_splits;
    std::vector<std::pair<long long, long long>> ranges;
    
    for (int i = 0; i < num_splits; ++i) {
        long long range_start = start + i * step;
        long long range_end = (i == num_splits - 1) ? end : range_start + step;
        ranges.push_back(std::make_pair(range_start, range_end));
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

    int num_threads = std::thread::hardware_concurrency();
    
    for (const auto& group : groups) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto ranges = split_range(group.first, group.second, num_threads);
        std::vector<std::future<std::pair<long long, int>>> futures;

        for (const auto& range : ranges) {
            futures.push_back(std::async(std::launch::async, calculate_collatz_range, range.first, range.second));
        }
        
        long long number_with_max_steps = 0;
        int max_steps = 0;

        for (auto& future : futures) {
            auto result = future.get();
            if (result.second > max_steps) {
                number_with_max_steps = result.first;
                max_steps = result.second;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Range " << group.first << " to " << group.second << ":\n";
        std::cout << "  Number with max steps: " << number_with_max_steps << " (" << max_steps << " steps)\n";
        std::cout << "Computation time: " << elapsed.count() << " seconds\n\n";
    }

    return 0;
}