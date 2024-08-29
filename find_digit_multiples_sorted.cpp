#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <unordered_set>

std::mutex results_mutex;

struct Result {
    int number;
    int multiplier;
    int digit_sum;
};

// Function to calculate the sum of digits of a number
int sum_of_digits(int num) {
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum;
}

// Function that each thread will run to find numbers in its range
void find_numbers(int start, int end, std::vector<Result>& results, std::unordered_set<int>& used_multipliers) {
    std::vector<Result> local_results;
    std::unordered_set<int> local_used_multipliers;
    
    for (int i = start; i <= end; i++) {
        int digit_sum = sum_of_digits(i);
        // Check for multiples from 2 to 1000000
        for (int k = 2; k <= 1000000; k++) {
            if (i == k * digit_sum) {
                local_results.push_back({i, k, digit_sum});
                local_used_multipliers.insert(k);
                break;
            }
        }
    }

    // Lock the results vector and add local results to the shared results
    std::lock_guard<std::mutex> guard(results_mutex);
    results.insert(results.end(), local_results.begin(), local_results.end());
    used_multipliers.insert(local_used_multipliers.begin(), local_used_multipliers.end());
}

int main() {
    int max_num = 1000000;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = max_num / num_threads;

    std::vector<std::thread> threads;
    std::vector<Result> results;
    std::unordered_set<int> used_multipliers;

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size + 1;
        int end = (i == num_threads - 1) ? max_num : (i + 1) * chunk_size;
        threads.emplace_back(find_numbers, start, end, std::ref(results), std::ref(used_multipliers));
    }

    for (auto& th : threads) {
        th.join();
    }

    // Sort results by multiplier (k) value
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        return a.multiplier < b.multiplier;
    });

    // Write the sorted results to the file
    std::ofstream outfile("find_digit_multiples.txt");

    for (const auto& res : results) {
        outfile << "Number: " << res.number << " is " << res.multiplier 
                << " times its digit sum (" << res.digit_sum << ")" << std::endl;
    }

    // Find and write unused multipliers to the file
    outfile << "\nUnused multipliers (k):\n";
    bool first = true;
    for (int k = 2; k <= 1000000; ++k) {
        if (used_multipliers.find(k) == used_multipliers.end()) {
            if (!first) {
                outfile << ", ";
            }
            outfile << k;
            first = false;
        }
    }
    outfile << std::endl;

    outfile.close();

    return 0;
}
