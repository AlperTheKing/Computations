#include <iostream>
#include <gmpxx.h>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

// Mutex for synchronized console output
std::mutex cout_mutex;

// Function to compute E(n) and check if it's an integer
void compute_E(int n) {
    auto start_time = std::chrono::high_resolution_clock::now();

    mpz_class numerator, denominator, result;
    mpz_class nn = n;
    mpz_class nn_pow_n2, nn_pow_n;

    // Compute n^(n^2)
    mpz_pow_ui(nn_pow_n2.get_mpz_t(), nn.get_mpz_t(), n * n);

    // Compute n^n
    mpz_pow_ui(nn_pow_n.get_mpz_t(), nn.get_mpz_t(), n);

    // numerator = n^(n^2) + 1
    numerator = nn_pow_n2 + 1;

    // denominator = n^n + 1
    denominator = nn_pow_n + 1;

    // Check if numerator is divisible by denominator
    if (mpz_divisible_p(numerator.get_mpz_t(), denominator.get_mpz_t())) {
        mpz_divexact(result.get_mpz_t(), numerator.get_mpz_t(), denominator.get_mpz_t());

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::lock_guard<std::mutex> guard(cout_mutex);
        std::cout << "n = " << n << " ("
                  << (n % 2 == 1 ? "odd" : "even") << "): E(n) is integer." << std::endl;
        std::cout << "E(" << n << ") = " << result.get_str() << std::endl;
        std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl << std::endl;
    } else {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::lock_guard<std::mutex> guard(cout_mutex);
        std::cout << "n = " << n << " ("
                  << (n % 2 == 1 ? "odd" : "even") << "): E(n) is not integer." << std::endl;
        std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl << std::endl;
    }
}

int main() {
    const int max_n = 100; // You can increase this value as needed
    std::vector<std::thread> threads;

    // Start time measurement
    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (int n = 1; n <= max_n; ++n) {
        threads.emplace_back(compute_E, n);
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // End time measurement
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

    std::cout << "Total execution time: " << total_elapsed.count() << " seconds." << std::endl;

    return 0;
}