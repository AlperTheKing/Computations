#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdint>

const unsigned long long MAX_LIMIT = 1000000000000ULL; // 1 trillion
const unsigned long long CHUNK_SIZE = 1000000000ULL; // 1 billion

// Function to perform the Sieve of Eratosthenes up to a given limit
void simple_sieve(unsigned long long limit, std::vector<bool>& is_prime) {
    for (unsigned long long i = 2; i * i <= limit; ++i) {
        if (is_prime[i]) {
            for (unsigned long long j = i * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

// Function to process chunks
void process_chunk(unsigned long long start, unsigned long long end, const std::vector<unsigned long long>& primes, std::ofstream& outfile) {
    std::vector<uint8_t> is_prime(end - start + 1, 1); // 1 means prime

    // Mark non-primes in the range
    for (const auto& prime : primes) {
        if (prime * prime > end) break;

        // Find the starting point in the chunk
        unsigned long long start_index = std::max(prime * prime, (start + prime - 1) / prime * prime);
        for (unsigned long long j = start_index; j <= end; j += prime) {
            is_prime[j - start] = 0;
        }
    }

    // Write primes to file
    for (unsigned long long i = start; i <= end; ++i) {
        if (is_prime[i - start] && i != 1) { // Ensure 1 is not included as prime
            outfile << i << " ";
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate memory for the initial small sieve
    std::vector<bool> small_sieve(std::sqrt(MAX_LIMIT) + 1, true);
    small_sieve[0] = small_sieve[1] = false;

    // Perform the Sieve of Eratosthenes on the small range
    simple_sieve(std::sqrt(MAX_LIMIT), small_sieve);

    // Store the primes up to sqrt(MAX_LIMIT)
    std::vector<unsigned long long> small_primes;
    for (unsigned long long i = 2; i <= std::sqrt(MAX_LIMIT); ++i) {
        if (small_sieve[i]) {
            small_primes.push_back(i);
        }
    }

    // Process chunks
    std::ofstream outfile("Primes1T.txt");
    for (unsigned long long start = 2; start <= MAX_LIMIT; start += CHUNK_SIZE) {
        unsigned long long end = std::min(start + CHUNK_SIZE - 1, MAX_LIMIT);
        process_chunk(start, end, small_primes, outfile);
    }
    outfile.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}