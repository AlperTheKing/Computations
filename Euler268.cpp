#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace std;
using namespace std::chrono;

// Function to compute the greatest common divisor (GCD) of two numbers
uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Function to compute the least common multiple (LCM) of two numbers with overflow check
bool safe_lcm(uint64_t a, uint64_t b, uint64_t N, uint64_t& result) {
    uint64_t g = gcd(a, b);
    uint64_t temp = a / g;
    if (temp > N / b) {
        // Overflow would occur
        return false;
    }
    result = temp * b;
    return true;
}

// Function to compute binomial coefficients C(n, k)
uint64_t binomial_coefficient(int n, int k) {
    if (k < 0 || k > n) return 0;
    uint64_t result = 1;
    for (int i = 1; i <= k; ++i) {
        result *= (n - k + i);
        result /= i;
    }
    return result;
}

// Function to print uint64_t numbers
void print_uint64(uint64_t n) {
    cout << n;
}

// Worker function for each thread
void worker(int thread_id, int num_threads, const uint64_t N, const vector<uint64_t>& primes, const int num_primes, const int t, const uint64_t* binom_coeff, uint64_t& partial_result) {
    uint64_t local_result = 0;

    // Calculate the total number of masks
    uint64_t total_masks = 1ULL << num_primes;

    // Determine the range of masks for this thread
    uint64_t masks_per_thread = total_masks / num_threads;
    uint64_t start_mask = thread_id * masks_per_thread;
    uint64_t end_mask = (thread_id == num_threads - 1) ? total_masks : start_mask + masks_per_thread;

    // Inclusion-Exclusion Principle
    for (uint64_t mask = start_mask; mask < end_mask; ++mask) {
        int bits = __builtin_popcountll(mask);
        if (bits < t) continue; // Skip subsets with fewer than t primes

        uint64_t multiple = 1;
        bool overflow = false;

        // Calculate the LCM of the primes in the current subset with overflow check
        for (int i = 0; i < num_primes; ++i) {
            if (mask & (1ULL << i)) {
                uint64_t p = primes[i];
                if (!safe_lcm(multiple, p, N, multiple)) {
                    overflow = true;
                    break;
                }
            }
        }

        if (overflow || multiple > N || multiple == 0) continue;

        uint64_t count = (N - 1) / multiple;

        // Correct sign and binomial coefficient based on Inclusion-Exclusion Principle
        int64_t sign = ((bits - t) % 2 == 0) ? 1 : -1;
        uint64_t coeff = binom_coeff[bits];
        local_result += sign * coeff * count;
    }

    // Store the local result in the partial result variable
    partial_result = local_result;
}

int main() {
    // Start measuring time
    auto start = high_resolution_clock::now();

    const uint64_t N = 10000000000000000ULL; // 10^16
    const vector<uint64_t> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                                    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    const int num_primes = primes.size();
    const int t = 4; // Minimum number of primes

    // Precompute binomial coefficients C(k - 1, t - 1) for k from t to num_primes
    uint64_t binom_coeff[26] = {0}; // Since k can be up to 25
    for (int k = t; k <= num_primes; ++k) {
        binom_coeff[k] = binomial_coefficient(k - 1, t - 1);
    }

    // Determine the number of threads to use
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default to 4 threads if detection fails

    // Create a vector to hold the partial results from each thread
    vector<uint64_t> partial_results(num_threads, 0);

    // Create and launch threads
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i, num_threads, N, cref(primes), num_primes, t, binom_coeff, ref(partial_results[i]));
    }

    // Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }

    // Combine the partial results
    uint64_t result = 0;
    for (const auto& partial_result : partial_results) {
        result += partial_result;
    }

    cout << "Number of positive integers less than 10^16 divisible by at least four distinct primes less than 100: ";
    print_uint64(result);
    cout << endl;

    // End measuring time
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds." << endl;

    return 0;
}