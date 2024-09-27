#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <mutex>

std::mutex mtx; // Mutex for protecting shared data

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string BOLD_RED = "\033[1;31m";
const std::string BOLD_GREEN = "\033[1;32m";
const std::string BOLD_YELLOW = "\033[1;33m";
const std::string BOLD_BLUE = "\033[1;34m";

// Function to calculate the number of Collatz steps for a given number
unsigned long long collatz_steps(unsigned long long n) {
    unsigned long long steps = 0;
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

// Function to find the number with maximum Collatz steps in a given range
void find_max_collatz_steps_in_range(unsigned long long start, unsigned long long end, unsigned long long &number_with_max_steps, unsigned long long &max_steps) {
    unsigned long long local_max_steps = 0;
    unsigned long long local_number_with_max_steps = start;

    for (unsigned long long i = start; i < end; ++i) {
        unsigned long long steps = collatz_steps(i);
        if (steps > local_max_steps) {
            local_max_steps = steps;
            local_number_with_max_steps = i;
        }
    }

    // Protect shared data using mutex
    std::lock_guard<std::mutex> guard(mtx);
    if (local_max_steps > max_steps) {
        max_steps = local_max_steps;
        number_with_max_steps = local_number_with_max_steps;
    }
}

// Wrapper function to handle threading
void thread_task(unsigned long long start, unsigned long long end, unsigned long long &number_with_max_steps, unsigned long long &max_steps) {
    find_max_collatz_steps_in_range(start, end, number_with_max_steps, max_steps);
}

int main() {
    // Define the ranges, now including 10^11-10^12
    std::vector<std::pair<unsigned long long, unsigned long long>> ranges;
    unsigned long long base = 1;  // Start from 10^0
    for (int i = 0; i <= 11; ++i) {
        unsigned long long start = base;
        unsigned long long end = base * 10;
        ranges.push_back(std::make_pair(start, end));
        base = end;
    }

    // Loop through each range and find the number with the maximum Collatz steps
    for (const auto &range : ranges) {
        unsigned long long start = range.first;
        unsigned long long end = range.second;

        unsigned long long number_with_max_steps = start;
        unsigned long long max_steps = 0;

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Split the range into chunks for parallel processing
        unsigned long long num_threads = std::thread::hardware_concurrency(); // Use the number of available cores
        unsigned long long chunk_size = (end - start) / num_threads;

        std::vector<std::thread> threads;
        std::vector<unsigned long long> local_max_steps(num_threads);
        std::vector<unsigned long long> local_number_with_max_steps(num_threads);

        for (unsigned long long i = 0; i < num_threads; ++i) {
            unsigned long long chunk_start = start + i * chunk_size;
            unsigned long long chunk_end = (i == num_threads - 1) ? end : chunk_start + chunk_size;

            threads.push_back(std::thread(thread_task, chunk_start, chunk_end, std::ref(local_number_with_max_steps[i]), std::ref(local_max_steps[i])));
        }

        // Join the threads
        for (auto &th : threads) {
            th.join();
        }

        // Find the overall maximum from all threads
        for (unsigned long long i = 0; i < num_threads; ++i) {
            if (local_max_steps[i] > max_steps) {
                max_steps = local_max_steps[i];
                number_with_max_steps = local_number_with_max_steps[i];
            }
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> execution_time = end_time - start_time;

        // Display the result vertically with color
        std::cout << BOLD_BLUE << "Range " << start << " - " << end << ":\n"
                  << BOLD_GREEN << "Number with max steps: " << number_with_max_steps << "\n"
                  << BOLD_YELLOW << "Steps: " << max_steps << "\n"
                  << BOLD_RED << "Time taken: " << execution_time.count() << " seconds\n"
                  << RESET << "-----------------------------\n";
    }

    return 0;
}