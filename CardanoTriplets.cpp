#include <iostream>
#include <vector>
#include <pthread.h>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <atomic>

typedef unsigned long long ull;

// Atomic variable for thread-safe operations
std::atomic<ull> total_count(0); // Total number of Cardano Triplets

// Structure to pass data to threads
struct ThreadData {
    ull start_a;
    ull end_a;
    ull max_sum;
};

// Function to factorize n and store the exponents of prime factors using trial division
void factorize(ull n, std::vector<std::pair<ull, ull>>& factors) {
    for (ull i = 2; i * i <= n; ++i) {
        ull count = 0;
        while (n % i == 0) {
            n /= i;
            ++count;
        }
        if (count > 0)
            factors.emplace_back(i, count);
    }
    if (n > 1)
        factors.emplace_back(n, 1);
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
void generate_bc(const std::vector<std::pair<ull, ull>>& factors, size_t idx,
                 ull b, ull c, ull max_sum, ull a) {
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

// Function executed by each thread to find Cardano Triplets
void* find_cardano_triplets(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    ull start_a = data->start_a;
    ull end_a = data->end_a;
    ull max_sum = data->max_sum;

    for (ull a = start_a; a <= end_a; a += 3) { // a ≡ 2 mod 3
        ull N = (1 + a) * (1 + a) * (8 * a - 1);
        if (N % 27 != 0)
            continue;
        ull N_div = N / 27;

        // Factorize N_div using trial division
        std::vector<std::pair<ull, ull>> factors;
        factorize(N_div, factors);

        // Generate all possible (b, c) pairs
        generate_bc(factors, 0, 1, 1, max_sum, a);
    }

    return nullptr;
}

int main() {
    ull max_sum;
    std::cout << "Enter the maximum value for (a + b + c): ";
    std::cin >> max_sum;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the number of hardware threads available
    unsigned int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads == 0)
        num_threads = 4; // Default to 4 if unable to determine

    std::cout << "Number of threads: " << num_threads << std::endl;

    // Estimate MAX_A based on max_sum
    ull MAX_A = max_sum; // Conservative estimate

    ull range = MAX_A / num_threads + 1; // Ensure full coverage

    // Create pthreads
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);

    for (unsigned int i = 0; i < num_threads; ++i) {
        ull start_a = 2 + i * range;
        if (start_a % 3 != 2)
            start_a += (3 - (start_a % 3) + 2) % 3; // Adjust to a ≡ 2 mod 3
        ull end_a = std::min(start_a + range - 1, MAX_A);

        if (start_a > end_a)
            continue;

        thread_data[i].start_a = start_a;
        thread_data[i].end_a = end_a;
        thread_data[i].max_sum = max_sum;

        pthread_create(&threads[i], nullptr, find_cardano_triplets, &thread_data[i]);
    }

    // Join threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Total Cardano Triplets: " << total_count.load() << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}