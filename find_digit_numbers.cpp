#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

std::mutex print_mutex;

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
void find_numbers(int start, int end) {
    for (int i = start; i <= end; i++) {
        int digit_sum = sum_of_digits(i);
        // Check for multiples from 2 to 10000
        for (int k = 2; k <= 10000; k++) {
            if (i == k * digit_sum) {
                std::lock_guard<std::mutex> guard(print_mutex);
                std::cout << "Number: " << i << " is " << k << " times its digit sum (" << digit_sum << ")" << std::endl;
            }
        }
    }
}

int main() {
    int max_num = 1000000;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = max_num / num_threads;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size + 1;
        int end = (i == num_threads - 1) ? max_num : (i + 1) * chunk_size;
        threads.emplace_back(find_numbers, start, end);
    }

    for (auto& th : threads) {
        th.join();
    }

    return 0;
}