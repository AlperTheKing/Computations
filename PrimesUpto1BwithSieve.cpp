#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    const int limit = 1000000000;  // 1 billion
    std::vector<bool> isPrime(limit + 1, true);
    isPrime[0] = isPrime[1] = false;

    auto start = std::chrono::high_resolution_clock::now();

    int sqrtLimit = static_cast<int>(std::sqrt(limit));
    for (int i = 2; i <= sqrtLimit; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= limit; j += i) {
                isPrime[j] = false;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (isPrime[i]) {
            primes.push_back(i);
        }
    }

    // Print the first 10 primes
    std::cout << "First 10 primes: ";
    for (int i = 0; i < 10 && i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    // Print the last 10 primes
    std::cout << "Last 10 primes: ";
    for (int i = primes.size() - 10; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Total primes found: " << primes.size() << std::endl;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}