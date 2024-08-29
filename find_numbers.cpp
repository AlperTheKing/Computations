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

// Function to check if a number meets the condition
void find_numbers(int start, int end, std::vector<int>& results) {
    for (int i = start; i <= end; i++) {
        int digit_sum = sum_of_digits(i);
        if (i == 300 * digit_sum) {
            std::lock_guard<std::mutex> guard(print_mutex);
            results.push_back(i);
        }
    }
}

int main() {
    int max_num = 1000000000;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = max_num / num_threads;

    std::vector<std::thread> threads;
    std::vector<int> results;

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size + 1;
        int end = (i == num_threads - 1) ? max_num : (i + 1) * chunk_size;
        threads.emplace_back(find_numbers, start, end, std::ref(results));
    }

    for (auto& th : threads) {
        th.join();
    }

    for (int num : results) {
        std::cout << num << std::endl;
    }

    return 0;
}