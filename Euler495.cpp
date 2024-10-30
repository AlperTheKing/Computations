#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <thread>
#include <chrono>
#include <functional>

#define MOD 1000000007

// Function to generate all primes up to n using Sieve of Eratosthenes
std::vector<int> sieve_of_eratosthenes(int n) {
    std::vector<bool> is_prime(n + 1, true);
    std::vector<int> primes;
    is_prime[0] = is_prime[1] = false;
    
    for (int p = 2; p <= n; ++p) {
        if (is_prime[p]) {
            primes.push_back(p);
            for (int i = 2 * p; i <= n; i += p) {
                is_prime[i] = false;
            }
        }
    }
    return primes;
}

// Function to calculate the exponent of a prime p in n!
int prime_exponent_in_factorial(int n, int p) {
    int exponent = 0;
    int power = p;
    while (power <= n) {
        exponent += n / power;
        if (power > n / p) break;  // Avoid overflow
        power *= p;
    }
    return exponent;
}

// Generate all divisors based on prime factorization of n!
void generate_divisors(const std::map<int, int>& prime_factors, std::vector<int>& divisors) {
    divisors.push_back(1);  // Start with 1 as a divisor

    for (const auto& pair : prime_factors) {
        int prime = pair.first;
        int exponent = pair.second;

        std::vector<int> new_divisors;
        int current_power = 1;

        for (int i = 0; i <= exponent; ++i) {
            for (int d : divisors) {
                new_divisors.push_back(d * current_power);
            }
            current_power *= prime;
        }

        divisors = std::move(new_divisors);
    }
}

// Function executed by each thread to compute part of the result
void count_combinations_thread(const std::vector<int>& divisors, int k, int target, int start, int end, int& local_count) {
    for (int i = start; i < end; ++i) {
        std::vector<int> comb(k);  // Store divisor combination
        comb[0] = divisors[i];

        // Generate combinations iteratively, avoiding recursion
        for (int j = i + 1; j < divisors.size() && comb.size() < k; ++j) {
            comb[comb.size()] = divisors[j];
            int product = 1;
            for (int c : comb) product *= c;
            if (product == target) {
                local_count = (local_count + 1) % MOD;
            }
        }
    }
}

// Main function to calculate W(n!, k) using multithreading
int W(int n, int k) {
    // Step 1: Prime factorization of n!
    std::vector<int> primes = sieve_of_eratosthenes(n);
    std::map<int, int> prime_factors;

    // Calculate exponents of all primes in the prime factorization of n!
    for (int prime : primes) {
        int exponent = prime_exponent_in_factorial(n, prime);
        if (exponent > 0) {
            prime_factors[prime] = exponent;
        }
    }

    // Step 2: Generate divisors from the prime factorization
    std::vector<int> divisors;
    generate_divisors(prime_factors, divisors);

    // Step 3: Multithreading to calculate combinations
    int total_count = 0;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback if no thread count is provided

    std::vector<std::thread> threads;
    std::vector<int> local_counts(num_threads, 0);  // Local counts for each thread

    // Divide divisors among threads
    int chunk_size = divisors.size() / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? divisors.size() : start + chunk_size;

        // Launch a thread to handle part of the divisors
        threads.emplace_back(count_combinations_thread, std::ref(divisors), k, n, start, end, std::ref(local_counts[t]));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Sum up all local counts
    for (int count : local_counts) {
        total_count = (total_count + count) % MOD;
    }

    return total_count;  // Return the final result
}

int main() {
    int n = 10000; // Example input n = 10000!
    int k = 30;

    // Measure the time taken to compute W(n!, k)
    auto start = std::chrono::high_resolution_clock::now();

    int result = W(n, k);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    // Output the result and elapsed time
    std::cout << "W(" << n << "!, " << k << ") = " << result << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}