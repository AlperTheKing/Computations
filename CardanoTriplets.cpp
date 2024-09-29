#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>

typedef unsigned long long ull;

std::atomic<ull> total_count(0); // Total number of Cardano Triplets

// Function to generate a list of primes using the Sieve of Eratosthenes
std::vector<ull> sieve(ull max_n) {
    std::vector<bool> is_prime(max_n + 1, true);
    std::vector<ull> primes;
    is_prime[0] = is_prime[1] = false;

    for (ull i = 2; i <= max_n; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (ull j = i * i; j <= max_n; j += i)
                is_prime[j] = false;
        }
    }
    return primes;
}

// Function to factorize n using precomputed primes
void factorize(ull n, const std::vector<ull>& primes, std::vector<std::pair<ull, ull>>& factors) {
    for (ull prime : primes) {
        if (prime * prime > n)
            break;
        ull count = 0;
        while (n % prime == 0) {
            n /= prime;
            ++count;
        }
        if (count > 0)
            factors.emplace_back(prime, count);
    }
    if (n > 1)
        factors.emplace_back(n, 1); // n is prime
}

// Integer exponentiation
ull int_pow(ull base, ull exp) {
    ull result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// Recursive function to generate all possible (b, c) pairs
void generate_bc(const std::vector<std::pair<ull, ull>>& factors, size_t idx, ull b, ull c, ull max_sum, ull a) {
    if (idx == factors.size()) {
        if (a + b + c <= max_sum && b > 0 && c > 0) {
            total_count.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    // Distribute exponents between b^2 and c, exponents in b^2 must be even
    ull p = factors[idx].first;
    ull e = factors[idx].second;

    ull max_k = e / 2;
    for (ull k = 0; k <= max_k; ++k) {
        ull b_new = b * int_pow(p, k);
        ull c_new = c * int_pow(p, e - 2 * k);
        generate_bc(factors, idx + 1, b_new, c_new, max_sum, a);
    }
}

// Function to find Cardano Triplets in a given range of 'a'
void find_cardano_triplets(ull start_a, ull end_a, ull max_sum, const std::vector<ull>& primes) {
    for (ull a = start_a; a <= end_a; a += 3) { // a ≡ 2 mod 3
        ull N = (1 + a) * (1 + a) * (8 * a - 1);
        if (N % 27 != 0)
            continue;
        ull N_div = N / 27;

        // Factorize N_div using precomputed primes
        std::vector<std::pair<ull, ull>> factors;
        factorize(N_div, primes, factors);

        // Generate all possible (b, c) pairs
        generate_bc(factors, 0, 1, 1, max_sum, a);
    }
}

int main() {
    ull max_sum;
    std::cout << "Enter the maximum value for (a + b + c): ";
    std::cin >> max_sum;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate primes up to the square root of the largest possible N
    ull max_prime = static_cast<ull>(std::sqrt((1 + max_sum) * (1 + max_sum) * (8 * max_sum - 1) / 27));
    std::vector<ull> primes = sieve(max_prime);

    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4; // Default to 4 if hardware_concurrency cannot determine

    std::cout << "Number of threads: " << num_threads << std::endl;

    // Estimate MAX_A based on max_sum
    ull MAX_A = max_sum; // Conservative estimate

    ull range = MAX_A / num_threads + 1; // Ensure full coverage
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        ull start_a = 2 + i * range;
        if (start_a % 3 != 2)
            start_a += (3 - (start_a % 3) + 2) % 3; // Adjust to a ≡ 2 mod 3
        ull end_a = std::min(start_a + range - 1, MAX_A);

        if (start_a > end_a)
            continue;

        threads.emplace_back(find_cardano_triplets, start_a, end_a, max_sum, std::ref(primes));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Total Cardano Triplets: " << total_count.load() << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}