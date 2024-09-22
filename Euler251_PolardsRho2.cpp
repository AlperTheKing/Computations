#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <tuple>
#include <mutex>
#include <cstdint>
#include <algorithm>
#include <random>

// Mutex for thread-safe access to the triplets vector
std::mutex triplet_mutex;

// Function to compute the Greatest Common Divisor (GCD) using Euclidean algorithm
int64_t gcd_custom(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

// Function to perform modular multiplication (a * b) % mod safely
int64_t mulmod_custom(int64_t a, int64_t b, int64_t mod) {
    __int128 res = (__int128(a) * __int128(b)) % mod;
    return (int64_t)res;
}

// Function to perform modular exponentiation (base^exp) % mod
int64_t power_custom(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1)
            result = mulmod_custom(result, base, mod);
        base = mulmod_custom(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// Miller-Rabin Primality Test
bool is_prime_custom(int64_t n) {
    if (n < 2)
        return false;
    // Bases for deterministic Miller-Rabin for n < 2^64
    const int64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    int64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s += 1;
    }
    for (int64_t a : bases) {
        if (a >= n)
            continue;
        int64_t x = power_custom(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool cont_outer = false;
        for (int r = 1; r < s; r++) {
            x = mulmod_custom(x, x, n);
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

// Pollard's Rho Algorithm for Factorization
int64_t pollards_rho_custom(int64_t n) {
    if (n % 2 == 0)
        return 2;
    if (n % 3 == 0)
        return 3;
    if (n % 5 == 0)
        return 5;

    std::mt19937_64 mt_rand(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(1, n - 1);

    auto f = [&](int64_t x, int64_t c, int64_t mod) -> int64_t {
        return (mulmod_custom(x, x, mod) + c) % mod;
    };

    for (int i = 0; i < 20; ++i) { // Retry with different parameters
        int64_t c = dist(mt_rand);
        int64_t x = dist(mt_rand);
        int64_t y = x;
        int64_t d = 1;
        while (d == 1) {
            x = f(x, c, n);
            y = f(f(y, c, n), c, n);
            d = gcd_custom(std::abs(x - y), n);
        }
        if (d != n)
            return d;
    }
    return n;
}

// Function to factorize a number into its prime factors using Pollard's Rho and Miller-Rabin
void factorize_custom(int64_t n, std::vector<int64_t>& factors) {
    if (n == 1)
        return;
    if (is_prime_custom(n)) {
        factors.push_back(n);
        return;
    }
    int64_t d = pollards_rho_custom(n);
    factorize_custom(d, factors);
    factorize_custom(n / d, factors);
}

// Function to get prime factors along with their exponents
std::vector<std::pair<int64_t, int>> get_prime_factors_custom(int64_t n) {
    std::vector<int64_t> factors;
    factorize_custom(n, factors);
    std::sort(factors.begin(), factors.end());
    std::vector<std::pair<int64_t, int>> prime_factors;
    if (factors.empty())
        return prime_factors;
    int count = 1;
    int64_t current = factors[0];
    for (size_t i = 1; i < factors.size(); ++i) {
        if (factors[i] == current)
            count++;
        else {
            prime_factors.emplace_back(std::make_pair(current, count));
            current = factors[i];
            count = 1;
        }
    }
    prime_factors.emplace_back(std::make_pair(current, count));
    return prime_factors;
}

// Function to calculate b such that b^2 divides d
// It computes b as the product of p_i^{floor(e_i / 2)} for each prime factor p_i with exponent e_i
int64_t calculate_b_custom(int64_t d) {
    std::vector<std::pair<int64_t, int>> prime_factors = get_prime_factors_custom(d);
    int64_t b = 1;
    for (const auto& pf : prime_factors) {
        int exponent = pf.second / 2;
        for (int i = 0; i < exponent; ++i) {
            b *= pf.first;
        }
    }
    return b;
}

// Function to calculate triplets and store them in the shared triplets vector
void tripletCounter_custom(int64_t start_n, int64_t end_n, int64_t limit,
                           std::vector<std::tuple<int64_t, int64_t, int64_t>>& shared_triplets) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> local_triplets;

    for (int64_t n = start_n; n < end_n; ++n) {
        int64_t k = 3 * n + 2;
        int64_t a = k;

        // Calculate (2k - 1)/3 safely since k = 3n + 2
        // (2k - 1)/3 = (6n + 4 -1)/3 = (6n +3)/3 = 2n +1
        int64_t term = 2 * n + 1;
        int64_t d = k * k + static_cast<int64_t>(std::pow(term, 3));

        if (d <= 0)
            continue; // Skip invalid d

        // Calculate b such that b^2 divides d
        int64_t b = calculate_b_custom(d);
        if (b == 0)
            continue;

        // Now, calculate c = d / b^2
        int64_t b_squared = b * b;
        if (b_squared == 0)
            continue; // Prevent division by zero
        if (d % b_squared != 0)
            continue; // b^2 must divide d

        int64_t c = d / b_squared;

        // Check the sum constraint
        if (a + b + c <= limit) {
            local_triplets.emplace_back(std::make_tuple(a, b, c));
        }
    }

    // Lock and append local triplets to the shared triplets vector
    if (!local_triplets.empty()) {
        std::lock_guard<std::mutex> guard(triplet_mutex);
        shared_triplets.insert(shared_triplets.end(), local_triplets.begin(), local_triplets.end());
    }
}

int main() {
    const int64_t limit = 1000;
    const unsigned int numThreads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::thread> threads;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> triplets;

    // Measure start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the maximum value of n such that k = 3n + 2 <= limit
    // Since a + b + c <= limit, and b, c >=1, k <= limit - 2
    int64_t max_n = (limit - 2) / 3;

    // Divide the range of n among threads
    std::vector<std::pair<int64_t, int64_t>> thread_ranges;
    int64_t range_size = max_n / numThreads;
    int64_t remainder = max_n % numThreads;
    int64_t current_start = 0;

    for (unsigned int i = 0; i < numThreads; ++i) {
        int64_t current_end = current_start + range_size;
        if (i < remainder) {
            current_end += 1;
        }
        thread_ranges.emplace_back(std::make_pair(current_start, current_end));
        current_start = current_end;
    }

    // Launch threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back(tripletCounter_custom, thread_ranges[i].first, thread_ranges[i].second, limit, std::ref(triplets));
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

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