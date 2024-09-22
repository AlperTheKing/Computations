#include <iostream>
#include <gmp.h>
#include <gmpxx.h>
#include <thread>
#include <chrono>
#include <vector>
#include <tuple>
#include <mutex>
#include <fstream>
#include <sstream>
#include <cstdlib>

// Mutex for thread-safe access to the triplets vector
std::mutex triplet_mutex;

// Function to execute GMP-ECM for factorization
std::vector<mpz_class> ecm_factorization(const mpz_class& n) {
    std::vector<mpz_class> factors;

    // Write the number to a file
    std::ofstream numfile("number.txt");
    numfile << n.get_str() << std::endl;
    numfile.close();

    // Execute GMP-ECM to factor the number
    system("ecm -c 1 110000000 < number.txt > factors.txt");

    // Read the factors from the file
    std::ifstream factorfile("factors.txt");
    std::string line;
    while (std::getline(factorfile, line)) {
        std::istringstream iss(line);
        mpz_class factor;
        if (iss >> factor) {
            factors.push_back(factor);
        }
    }

    return factors;
}

// Function to factorize a number and compute b such that b^2 divides d
int64_t calculate_b(const mpz_class& d) {
    std::vector<mpz_class> factors = ecm_factorization(d);
    int64_t b = 1;

    // Compute b as the product of p_i^{floor(e_i / 2)} for each prime factor p_i
    for (const mpz_class& factor : factors) {
        b *= mpz_get_ui(factor.get_mpz_t()); // Convert to regular int64_t
    }

    return b;
}

// Function to calculate triplets and store them in the shared triplets vector
void tripletCounter(int64_t start_n, int64_t end_n, int64_t limit,
                   std::vector<std::tuple<int64_t, int64_t, int64_t>>& shared_triplets) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> local_triplets;

    for (int64_t n = start_n; n < end_n; ++n) {
        int64_t k = 3 * n + 2;
        int64_t a = k;

        // Calculate (2k - 1)/3 safely since k = 3n + 2
        // (2k - 1)/3 = (6n + 4 -1)/3 = (6n +3)/3 = 2n +1
        int64_t term = 2 * n + 1;
        mpz_class d = mpz_class(k) * mpz_class(k) + mpz_class(term) * mpz_class(term) * mpz_class(term);

        // Use ECM to factor d and compute b
        int64_t b = calculate_b(d);

        if (b == 0)
            continue;

        // Now, calculate c = d / b^2
        int64_t b_squared = b * b;
        if (b_squared == 0)
            continue; // Prevent division by zero
        if (mpz_divisible_ui_p(d.get_mpz_t(), b_squared) == 0)
            continue; // b^2 must divide d

        mpz_class c_mpz;
        mpz_divexact_ui(c_mpz.get_mpz_t(), d.get_mpz_t(), b_squared); // c = d / b^2
        int64_t c = mpz_get_ui(c_mpz.get_mpz_t());

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
    const int64_t limit = 110000000; // 110 million
    const unsigned int numThreads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::thread> threads;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> triplets;

    // Measure start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the maximum value of n such that k = 3n + 2 <= limit - 2 (since b and c are at least 1)
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
        threads.emplace_back(tripletCounter, thread_ranges[i].first, thread_ranges[i].second, limit, std::ref(triplets));
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print all triplets
    for (const auto& triplet : triplets) {
        int64_t a, b, c;
        std::tie(a, b, c) = triplet;
        std::cout << "Triplet: a = " << a << ", b = " << b << ", c = " << c << std::endl;
    }

    // Print summary
    std::cout << "Total number of triplets found = " << triplets.size() << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}