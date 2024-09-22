#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <tuple>
#include <mutex>
#include <cstdint>
#include <algorithm>
#include <numeric>  
#include <map>
#include <random123/philox.h>

// Mutex for thread-safe access to the triplets vector
std::mutex triplet_mutex;

// Function to perform modular multiplication (a * b) % mod using __int128 to prevent overflow
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    __int128 res = (__int128(a) * b) % mod;
    return static_cast<uint64_t>(res);
}

// Function to perform modular exponentiation (base^exp) % mod
uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            res = mulmod(res, base, mod);
        base = mulmod(base, base, mod);
        exp >>= 1;
    }
    return res;
}

// Miller-Rabin Primality Test
bool is_prime(uint64_t n) {
    if (n < 2)
        return false;
    // Base cases
    for (auto p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (n % p == 0) {
            return n == p;
        }
    }

    // Write n-1 as 2^s * d
    uint64_t d = n - 1;
    uint64_t s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s += 1;
    }

    // Witnesses for n < 2^64 from https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    uint64_t witnesses[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};

    for (auto a : witnesses) {
        if (a >= n)
            continue;
        uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool cont_outer = false;
        for (uint64_t r = 1; r < s; ++r) {
            x = mulmod(x, x, n);
            if (x == n - 1) {
                cont_outer = true;
                break;
            }
        }
        if (cont_outer)
            continue;
        return false;
    }
    return true;
}

// Pollard's Rho Algorithm using Random123 for random number generation
uint64_t pollards_rho(uint64_t n, r123::Philox4x32_R<10>::key_type& key, r123::Philox4x32_R<10>::ctr_type& ctr, uint64_t& counter) {
    if (n % 2 == 0)
        return 2;
    if (n % 3 == 0)
        return 3;
    if (n % 5 == 0)
        return 5;

    auto f = [&](uint64_t x, uint64_t c) -> uint64_t {
        return (mulmod(x, x, n) + c) % n;
    };

    r123::Philox4x32_R<10> philox;

    for (int i = 0; i < 20; ++i) { // Retry with different constants
        // Generate a random 'c' using Random123's Philox
        ctr[0] = counter++;
        auto rand_val = philox(ctr, key);  // Generate a random number

        // Manually scale to [0, 1) and then to [1, n-1]
        double rand_uniform = static_cast<double>(rand_val[0]) / std::numeric_limits<uint32_t>::max();
        uint64_t c = 1 + static_cast<uint64_t>(rand_uniform * (n - 1)); // Ensure 1 <= c <= n-1

        uint64_t x = 2;
        uint64_t y = 2;
        uint64_t d = 1;

        while (d == 1) {
            x = f(x, c);
            y = f(f(y, c), c);
            d = std::gcd(x > y ? x - y : y - x, n);
        }

        if (d != n)
            return d;
    }
    return n;
}

// Function to factorize n using Pollard's Rho and return prime factors
void factor(uint64_t n, std::vector<uint64_t>& factors, r123::Philox4x32_R<10>::key_type& key, r123::Philox4x32_R<10>::ctr_type& ctr, uint64_t& counter) {
    if (n == 1)
        return;
    if (is_prime(n)) {
        factors.push_back(n);
        return;
    }
    uint64_t d = pollards_rho(n, key, ctr, counter);
    factor(d, factors, key, ctr, counter);
    factor(n / d, factors, key, ctr, counter);
}

// Function to generate all possible b values such that b^2 divides d
void generate_b_values(uint64_t d, std::vector<uint64_t>& b_values, r123::Philox4x32_R<10>::key_type& key, r123::Philox4x32_R<10>::ctr_type& ctr, uint64_t& counter) {
    std::vector<uint64_t> factors;
    factor(d, factors, key, ctr, counter);
    if (factors.empty()) {
        b_values.push_back(1);
        return;
    }

    // Count the exponents of each prime factor
    std::map<uint64_t, int> prime_counts;
    for (auto p : factors)
        prime_counts[p] += 1;

    // For b^2 to divide d, each exponent in b must be <= floor(e_p / 2)
    std::vector<std::pair<uint64_t, int>> primes;
    for (auto &[p, cnt] : prime_counts) {
        primes.emplace_back(p, cnt / 2);
    }

    // Generate all possible combinations of exponents
    std::vector<uint64_t> exponents(primes.size(), 0);
    size_t num_primes = primes.size();
    bool done = false;

    while (!done) {
        // Compute b based on current exponents
        uint64_t b = 1;
        for (size_t i = 0; i < num_primes; ++i) {
            for (int j = 0; j < exponents[i]; ++j)
                b *= primes[i].first;
        }
        b_values.push_back(b);

        // Increment exponents
        for (size_t i = 0; i < num_primes; ++i) {
            if (exponents[i] < primes[i].second) {
                exponents[i]++;
                break;
            }
            else {
                exponents[i] = 0;
                if (i == num_primes - 1)
                    done = true;
            }
        }
    }

    // Ensure that 1 is included
    if (b_values.empty())
        b_values.push_back(1);
}

// Function to calculate triplets and store them in the shared triplets vector
void tripletCounter(int64_t start_n, int64_t end_n, int64_t limit,
                   std::vector<std::tuple<int64_t, int64_t, int64_t>>& shared_triplets,
                   r123::Philox4x32_R<10>::key_type key_init, r123::Philox4x32_R<10>::ctr_type ctr_init, uint64_t counter_init) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> local_triplets;
    r123::Philox4x32_R<10>::key_type key = key_init;
    r123::Philox4x32_R<10>::ctr_type ctr = ctr_init;
    uint64_t counter = counter_init;

    for (int64_t n = start_n; n < end_n; ++n) {
        int64_t k = 3 * n + 2;
        int64_t a = k;

        // Calculate d = k^2 + ((2k - 1)/3)^3
        // Since k = 3n + 2, (2k -1)/3 = 2n +1
        int64_t term = 2 * n + 1;
        uint64_t d = static_cast<uint64_t>(k) * k + static_cast<uint64_t>(term) * term * term;

        // Generate all possible b values such that b^2 divides d
        std::vector<uint64_t> b_values;
        generate_b_values(d, b_values, key, ctr, counter);

        for (auto b : b_values) {
            uint64_t b_squared = static_cast<uint64_t>(b) * b;
            if (b_squared == 0)
                continue;
            if (d % b_squared != 0)
                continue; // Just in case, though should be guaranteed
            int64_t c = static_cast<int64_t>(d / b_squared);
            if (a + static_cast<int64_t>(b) + c <= limit) {
                local_triplets.emplace_back(a, static_cast<int64_t>(b), c);
            }
        }
    }

    // Lock and append local triplets to the shared triplets vector
    if (!local_triplets.empty()) {
        std::lock_guard<std::mutex> guard(triplet_mutex);
        shared_triplets.insert(shared_triplets.end(), local_triplets.begin(), local_triplets.end());
    }
}

int main() {
    const int64_t limit = 110000000;
    const unsigned int numThreads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::thread> threads;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> triplets;

    // Initialize Random123 Philox key and counter for main thread
    r123::Philox4x32_R<10>::key_type key_init = {0xDEADBEEF, 0xCAFEBABE}; // Corrected to 2 elements
    r123::Philox4x32_R<10>::ctr_type ctr_init = {0, 0}; // Corrected to 2 elements
    uint64_t counter_init = 0;

    // Measure start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the maximum value of n such that k = 3n + 2
    // Given a + b + c <= limit, and a = 3n + 2, b and c are at least 1
    // So 3n + 2 + 1 + 1 <= limit => 3n <= limit - 4 => n <= (limit -4)/3
    int64_t max_n = (limit - 4) / 3;

    // Divide the range of n among threads
    std::vector<std::pair<int64_t, int64_t>> thread_ranges;
    int64_t range_size = max_n / numThreads;
    int64_t remainder = max_n % numThreads;
    int64_t current_start = 0;

    for (unsigned int i = 0; i < numThreads; ++i) {
        int64_t current_end = current_start + range_size;
        if (i < remainder)
            current_end += 1;
        thread_ranges.emplace_back(std::make_pair(current_start, current_end));
        current_start = current_end;
    }

    // Launch threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        // Each thread gets a unique key and counter
        r123::Philox4x32_R<10>::key_type thread_key = key_init;
        r123::Philox4x32_R<10>::ctr_type thread_ctr = ctr_init;
        uint64_t thread_counter = counter_init + i * 1000; // Offset counters to ensure uniqueness

        threads.emplace_back(tripletCounter, thread_ranges[i].first, thread_ranges[i].second, limit,
                             std::ref(triplets), thread_key, thread_ctr, thread_counter);
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // // Sort triplets for better readability (optional)
    // std::sort(triplets.begin(), triplets.end());

    // // Print all triplets
    // for (const auto& triplet : triplets) {
    //     int64_t a, b, c;
    //     std::tie(a, b, c) = triplet;
    //     std::cout << "Triplet: a = " << a << ", b = " << b << ", c = " << c << std::endl;
    // }

    // Print summary
    std::cout << "Total number of triplets found = " << triplets.size() << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}