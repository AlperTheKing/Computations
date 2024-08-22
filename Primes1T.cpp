#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

const unsigned long long MAX_LIMIT = 1000000000000ULL;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate memory for the sieve
    std::vector<bool> is_prime(MAX_LIMIT + 1, true);
    is_prime[0] = is_prime[1] = false;

    // Perform the Sieve of Eratosthenes
    for (unsigned long long i = 2; i * i <= MAX_LIMIT; i++) {
        if (is_prime[i]) {
            for (unsigned long long j = i * i; j <= MAX_LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // Store the primes
    std::vector<unsigned long long> primes;
    for (unsigned long long i = 2; i <= MAX_LIMIT; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Output primes to file
    std::ofstream outfile("Primes1T.txt");
    for (const auto& prime : primes) {
        outfile << prime << " ";
    }
    outfile.close();

    // Print the first 30 and last 30 primes
    std::cout << "First 30 primes:" << std::endl;
    for (size_t i = 0; i < 30 && i < primes.size(); i++) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last 30 primes:" << std::endl;
    for (size_t i = primes.size() > 30 ? primes.size() - 30 : 0; i < primes.size(); i++) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}