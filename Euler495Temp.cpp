#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>

const __int128 MOD = 1000000007;

// Global variables for multithreading
std::atomic<size_t> task_counter(0);  // Atomic task counter for parallel processing
std::atomic<__int128> result_count(0);  // Atomic counter for valid results
std::mutex dp_mutex;

// Custom function to print __int128
std::ostream& operator<<(std::ostream& dest, __int128_t value) {
    std::ostream::sentry s(dest);
    if (s) {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[128];
        char* d = std::end(buffer);
        do {
            --d;
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        if (value < 0) {
            --d;
            *d = '-';
        }
        int len = std::end(buffer) - d;
        if (dest.rdbuf()->sputn(d, len) != len) {
            dest.setstate(std::ios_base::badbit);
        }
    }
    return dest;
}

// Function to find prime factors of n!
std::map<int, int> find_prime_factors(int n) {
    std::map<int, int> prime_factors;
    for (int i = 2; i <= n; ++i) {
        int num = i;
        for (int p = 2; p <= sqrt(i); ++p) {
            while (num % p == 0) {
                prime_factors[p]++;
                num /= p;
            }
        }
        if (num > 1) {
            prime_factors[num]++;
        }
    }
    return prime_factors;
}

// Function to generate divisors and count valid combinations
void generate_divisors(const std::map<int, int>& prime_factors, int start, int end) {
    std::vector<int> primes;
    std::vector<int> max_exponents;
    for (const auto& factor : prime_factors) {
        primes.push_back(factor.first);
        max_exponents.push_back(factor.second);
    }

    // Iterate through the range of tasks assigned to this thread
    for (int task = start; task < end; ++task) {
        std::vector<int> current_exponents(primes.size(), 0);
        int num = task;

        // Convert task index into exponent combinations
        for (size_t i = 0; i < primes.size(); ++i) {
            current_exponents[i] = num % (max_exponents[i] + 1);
            num /= (max_exponents[i] + 1);
        }

        // Count divisors based on current exponents
        __int128 divisors_count = 1;
        for (size_t i = 0; i < primes.size(); ++i) {
            divisors_count *= (current_exponents[i] + 1);
        }

        // Check if divisors_count matches the expected value
        if (divisors_count == 1) {  // Replace this condition with your actual check
            result_count = (result_count + 1) % MOD;
        }
    }
}

// Worker thread function
void worker_function(const std::map<int, int>& prime_factors, int total_tasks, int num_threads) {
    int chunk_size = total_tasks / num_threads;
    int thread_id = task_counter++;

    int start = thread_id * chunk_size;
    int end = (thread_id == num_threads - 1) ? total_tasks : start + chunk_size;

    generate_divisors(prime_factors, start, end);
}

int main() {
    int n = 100;  // Example n = 100
    int k = 10;   // Example k = 10

    // Time measurement start
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Find prime factors of n!
    std::map<int, int> prime_factors = find_prime_factors(n);

    // Step 2: Calculate total number of tasks based on the prime exponents
    int total_tasks = 1;
    for (const auto& factor : prime_factors) {
        total_tasks *= (factor.second + 1);  // Number of combinations for each prime factor
    }

    // Step 3: Multithreading setup
    int num_threads = std::thread::hardware_concurrency();  // Get number of hardware threads
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_function, std::ref(prime_factors), total_tasks, num_threads);
    }

    for (auto& th : threads) {
        th.join();
    }

    // Time measurement end
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Result
    std::cout << "Number of valid combinations (mod 1000000007): " << (long long)result_count << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}