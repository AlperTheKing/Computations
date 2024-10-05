#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;
using cpp_int = boost::multiprecision::cpp_int;

// Precompute factorials of digits 0 to 9
const std::vector<cpp_int> factorials = []() {
    std::vector<cpp_int> f(10, 1);
    for (int i = 1; i <= 9; ++i)
        f[i] = f[i - 1] * i;
    return f;
}();

const int MAX_I = 150; // Maximum value of i to compute
std::vector<cpp_int> g_values(MAX_I + 1, 0);
std::vector<int> sg_values(MAX_I + 1, 0);
std::mutex mtx;

void compute_g_sg(int start_d, int end_d) {
    // dp[sum_fact] = minimal number n as string
    std::unordered_map<cpp_int, std::string> dp;
    dp[0] = ""; // Base case

    // Maximum number of digits to consider
    const int max_digits = 50; // Adjusted to handle larger numbers

    // Iterate over the number of digits
    for (int digits = 1; digits <= max_digits; ++digits) {
        std::unordered_map<cpp_int, std::string> new_dp;

        // Iterate over possible digits (0 to 9)
        for (int d = start_d; d <= end_d; ++d) {
            cpp_int fact_d = factorials[d];

            for (const auto& [sum_fact, num_str] : dp) {
                cpp_int new_sum_fact = sum_fact + fact_d;
                std::string new_num_str = num_str + std::to_string(d);

                // Compute sf(n): sum of digits of new_sum_fact
                cpp_int temp_sum_fact = new_sum_fact;
                int sf_n = 0;
                while (temp_sum_fact > 0) {
                    sf_n += static_cast<int>(temp_sum_fact % 10);
                    temp_sum_fact /= 10;
                }

                // Skip if sf_n exceeds MAX_I
                if (sf_n > MAX_I)
                    continue;

                // Update g_values and sg_values
                cpp_int n = cpp_int(new_num_str);
                std::lock_guard<std::mutex> lock(mtx);
                if (g_values[sf_n] == 0 || n < g_values[sf_n]) {
                    g_values[sf_n] = n;
                    // Compute sg(i): sum of digits of n
                    int sg_i = 0;
                    for (char c : new_num_str)
                        sg_i += c - '0';
                    sg_values[sf_n] = sg_i;
                }

                // Update new_dp
                if (new_dp.find(new_sum_fact) == new_dp.end() || new_dp[new_sum_fact] > new_num_str) {
                    new_dp[new_sum_fact] = new_num_str;
                }
            }
        }

        // Use new_dp for the next iteration
        dp = std::move(new_dp);
    }
}

int main() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Threading setup
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;

    // Split the digit range between threads
    int digits_per_thread = 9 / num_threads;
    int remainder = 9 % num_threads;

    int current_digit = 0;
    for (int t = 0; t < num_threads; ++t) {
        int start_digit = current_digit;
        int end_digit = start_digit + digits_per_thread - 1;
        if (remainder > 0) {
            end_digit++;
            remainder--;
        }
        if (end_digit > 9) end_digit = 9;
        threads.emplace_back(compute_g_sg, start_digit, end_digit);
        current_digit = end_digit + 1;
    }

    for (auto& th : threads) {
        th.join();
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    // Calculate and print total sg(i)
    cpp_int total_sg = 0;
    for (int i = 1; i <= MAX_I; ++i) {
        if (g_values[i] != 0) {
            total_sg += sg_values[i];
            std::cout << "g(" << i << ") = " << g_values[i] << ", sg(" << i << ") = " << sg_values[i] << std::endl;
        } else {
            std::cout << "g(" << i << ") not found." << std::endl;
        }
    }

    std::cout << "\nTotal execution time: " << execution_time.count() << " seconds" << std::endl;
    std::cout << "The sum of sg(i) for 1 <= i <= " << MAX_I << " is: " << total_sg << std::endl;

    return 0;
}