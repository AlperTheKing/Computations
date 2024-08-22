#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    const long long MAX_LIMIT = 1000000000;  // 1 billion
    std::vector<bool> is_prime(MAX_LIMIT + 1, true);
    is_prime[0] = is_prime[1] = false;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Sieve of Eratosthenes
    for (long long i = 2; i * i <= MAX_LIMIT; ++i) {
        if (is_prime[i]) {
            for (long long j = i * i; j <= MAX_LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Collecting all primes
    std::vector<long long> primes;
    for (long long i = 2; i <= MAX_LIMIT; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }

    // Output the timing information
    std::cout << "Time taken to find all primes up to 1 billion: " << duration.count() << " seconds\n";

    // Print the first 10 and last 10 primes
    std::cout << "First 10 primes:\n";
    for (size_t i = 0; i < 10 && i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Last 10 primes:\n";
    for (size_t i = primes.size() > 10 ? primes.size() - 10 : 0; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Total number of primes found: " << primes.size() << "\n";

    return 0;
}