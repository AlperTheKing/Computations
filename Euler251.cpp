#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <set>
#include <tuple>
#include <string>
#include <sstream>

typedef unsigned long long ull;
typedef __uint128_t ull128; // Use 128-bit integers for large numbers

std::mutex mtx;
std::set<std::tuple<ull128, ull128, ull128>> triplet_set; // Set to store unique triplets

// Custom output function for __int128_t
std::ostream& operator<<(std::ostream& dest, __uint128_t value) {
    std::ostream::sentry s(dest);
    if (s) {
        __uint128_t tmp = value;
        char buffer[128];
        char* d = std::end(buffer);
        do {
            --d;
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        dest.write(d, std::end(buffer) - d);
    }
    return dest;
}

// Custom input function for __int128_t
std::istream& operator>>(std::istream& is, __uint128_t& value) {
    std::string str;
    is >> str;
    value = 0;
    for (char c : str) {
        if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
        } else {
            is.setstate(std::ios::failbit); // Set error state if invalid character found
            break;
        }
    }
    return is;
}

// Function to generate a list of primes using the Sieve of Eratosthenes
std::vector<ull128> sieve(ull128 max_n) {
    std::vector<bool> is_prime(static_cast<size_t>(max_n + 1), true);
    std::vector<ull128> primes;
    is_prime[0] = is_prime[1] = false;

    for (ull128 i = 2; i <= max_n; ++i) {
        if (is_prime[static_cast<size_t>(i)]) {
            primes.push_back(i);
            for (ull128 j = i * i; j <= max_n; j += i)
                is_prime[static_cast<size_t>(j)] = false;
        }
    }
    return primes;
}

// Function to factorize n using precomputed primes
void factorize(ull128 n, const std::vector<ull128>& primes, std::vector<std::pair<ull128, ull128>>& factors) {
    for (ull128 prime : primes) {
        if (prime * prime > n)
            break;
        ull128 count = 0;
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
ull128 int_pow(ull128 base, ull128 exp) {
    ull128 result = 1;
    while (exp > 0) {
        if (exp & 1)
            result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

// Manual integer square root calculation for 128-bit integers
ull128 int_sqrt(ull128 x) {
    ull128 left = 0, right = x;
    while (left < right) {
        ull128 mid = (left + right + 1) / 2;
        if (mid * mid <= x)
            left = mid;
        else
            right = mid - 1;
    }
    return left;
}

// Recursive function to generate all possible (b, c) pairs
void generate_bc(const std::vector<std::pair<ull128, ull128>>& factors, size_t idx, ull128 b, ull128 c, ull128 max_sum, ull128 a) {
    if (idx == factors.size()) {
        if (a + b + c <= max_sum && b > 0 && c > 0) {
            // Ensure b <= c to avoid duplicates
            if (b > c) std::swap(b, c);

            // Insert triplet into the set
            std::lock_guard<std::mutex> lock(mtx);
            triplet_set.insert(std::make_tuple(a, b, c));
        }
        return;
    }

    // Distribute exponents between b^2 and c, exponents in b^2 must be even
    ull128 p = factors[idx].first;
    ull128 e = factors[idx].second;

    ull128 max_k = e / 2;
    for (ull128 k = 0; k <= max_k; ++k) {
        ull128 b_new = b * int_pow(p, k);
        ull128 c_new = c * int_pow(p, e - 2 * k);
        generate_bc(factors, idx + 1, b_new, c_new, max_sum, a);
    }
}

// Function to find Cardano Triplets in a given range of 'a'
void find_cardano_triplets(ull128 start_a, ull128 end_a, ull128 max_sum, const std::vector<ull128>& primes) {
    for (ull128 a = start_a; a <= end_a; a += 3) { // a ≡ 2 mod 3
        ull128 N = (1 + a) * (1 + a) * (8 * a - 1);
        if (N % 27 != 0)
            continue;
        ull128 N_div = N / 27;

        // Factorize N_div using precomputed primes
        std::vector<std::pair<ull128, ull128>> factors;
        factorize(N_div, primes, factors);

        // Generate all possible (b, c) pairs
        generate_bc(factors, 0, 1, 1, max_sum, a);
    }
}

int main() {
    ull128 max_sum;
    std::cout << "Enter the maximum value for (a + b + c): ";
    std::cin >> max_sum;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate primes up to the square root of the largest possible N
    ull128 max_prime = int_sqrt((1 + max_sum) * (1 + max_sum) * (8 * max_sum - 1) / 27);
    std::vector<ull128> primes = sieve(max_prime);

    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4; // Default to 4 if hardware_concurrency cannot determine

    std::cout << "Number of threads: " << num_threads << std::endl;

    // Estimate MAX_A based on max_sum
    ull128 MAX_A = max_sum; // Conservative estimate

    ull128 range = MAX_A / num_threads + 1; // Ensure full coverage
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        ull128 start_a = 2 + i * range;
        if (start_a % 3 != 2)
            start_a += (3 - (start_a % 3) + 2) % 3; // Adjust to a ≡ 2 mod 3
        ull128 end_a = std::min(start_a + range - 1, MAX_A);

        if (start_a > end_a)
            continue;

        threads.emplace_back(find_cardano_triplets, start_a, end_a, max_sum, std::ref(primes));
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Output the total number of unique Cardano triplets found
    std::cout << "Total Cardano Triplets: " << triplet_set.size() << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}