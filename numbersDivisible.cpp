#include <iostream>
#include <gmpxx.h>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

// Mutex for synchronized console output
std::mutex cout_mutex;

// Function to perform Miller-Rabin primality test
bool is_prime(const mpz_class& n, int iterations = 25) {
    // Handle trivial cases
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (mpz_divisible_ui_p(n.get_mpz_t(), 2)) return false;

    mpz_class d = n - 1;
    unsigned long s = 0;

    // Factor out powers of 2 from n - 1
    while (mpz_divisible_ui_p(d.get_mpz_t(), 2)) {
        d /= 2;
        ++s;
    }

    gmp_randclass rng(gmp_randinit_mt);
    rng.seed(time(nullptr));

    for (int i = 0; i < iterations; ++i) {
        // Generate random integer a in [2, n - 2]
        mpz_class a = rng.get_z_range(n - 3) + 2;

        mpz_class x;
        mpz_powm(x.get_mpz_t(), a.get_mpz_t(), d.get_mpz_t(), n.get_mpz_t());

        if (x == 1 || x == n - 1) continue;

        bool continue_outer = false;
        for (unsigned long r = 1; r < s; ++r) {
            mpz_powm_ui(x.get_mpz_t(), x.get_mpz_t(), 2, n.get_mpz_t());
            if (x == n - 1) {
                continue_outer = true;
                break;
            }
        }

        if (continue_outer) continue;

        return false; // Composite
    }

    return true; // Probably prime
}

// Function to compute properties of N(n)
void compute_N(int n) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Construct the number N(n)
    std::string pattern = "10";
    std::string number_str;
    for (int i = 0; i < n; ++i) {
        number_str += pattern;
    }
    number_str += "1"; // Append "1" at the end

    // Convert string to mpz_class
    mpz_class N(number_str);

    // Results
    bool divisible_by_101 = mpz_divisible_ui_p(N.get_mpz_t(), 101);
    bool divisible_by_3 = mpz_divisible_ui_p(N.get_mpz_t(), 3);
    bool divisible_by_11 = mpz_divisible_ui_p(N.get_mpz_t(), 11);
    bool divisible_by_7 = mpz_divisible_ui_p(N.get_mpz_t(), 7);
    bool prime = is_prime(N);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    std::lock_guard<std::mutex> guard(cout_mutex);
    std::cout << "N(" << n << ") = " << N.get_str() << std::endl;
    std::cout << "Divisible by 101: " << (divisible_by_101 ? "Yes" : "No") << std::endl;
    std::cout << "Divisible by 3: " << (divisible_by_3 ? "Yes" : "No") << std::endl;
    std::cout << "Divisible by 11: " << (divisible_by_11 ? "Yes" : "No") << std::endl;
    std::cout << "Divisible by 7: " << (divisible_by_7 ? "Yes" : "No") << std::endl;
    std::cout << "Prime: " << (prime ? "Yes" : "No") << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds." << std::endl << std::endl;
}

int main() {
    const int max_n = 100; // You can adjust this value as needed
    std::vector<std::thread> threads;

    // Start total time measurement
    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (int n = 1; n <= max_n; ++n) {
        threads.emplace_back(compute_N, n);
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // End total time measurement
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

    std::cout << "Total execution time: " << total_elapsed.count() << " seconds." << std::endl;

    return 0;
}