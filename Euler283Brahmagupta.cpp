#include <gmp.h>
#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

std::mutex sum_mutex;
mpz_t total_sum;

void compute_heronian(int max_ratio, int start, int end) {
    mpz_t a, b, c, perimeter, s_p, area_squared, area, temp1, temp2;
    mpz_inits(a, b, c, perimeter, s_p, area_squared, area, temp1, temp2, NULL);

    for (int p = start; p <= end; ++p) {
        for (int w1 = 1; w1 <= 18; ++w1) {
            for (int s = 1; s <= 18; ++s) {
                for (int t = 1; t <= 18; ++t) {
                    for (int u = 1; u <= 18; ++u) {
                        for (int v = 1; v <= 18; ++v) {
                            for (int alpha = 1; alpha <= 18; ++alpha) {
                                for (int beta = 1; beta <= 18; ++beta) {
                                    for (int gamma = 1; gamma <= 18; ++gamma) {
                                        // Calculate a, b, c (sides of the triangle)
                                        mpz_set_ui(a, p);
                                        mpz_mul_ui(a, a, alpha * u * ((beta * w1 * v) * (beta * w1 * v) + (gamma * s * t) * (gamma * s * t)));

                                        mpz_set_ui(b, p);
                                        mpz_mul_ui(b, b, beta * s * ((alpha * w1 * t) * (alpha * w1 * t) + (gamma * u * v) * (gamma * u * v)));

                                        mpz_set_ui(c, p);
                                        mpz_mul_ui(c, c, (beta * u * v * v + alpha * s * t * t) * (beta * alpha * w1 * w1 - gamma * gamma * s * u));

                                        // Calculate perimeter
                                        mpz_add(perimeter, a, b);
                                        mpz_add(perimeter, perimeter, c);

                                        // Calculate semi-perimeter
                                        mpz_fdiv_q_ui(s_p, perimeter, 2);

                                        // Calculate area using Heron's formula
                                        mpz_sub(temp1, s_p, a); // temp1 = s_p - a
                                        mpz_sub(temp2, s_p, b); // temp2 = s_p - b
                                        mpz_mul(area_squared, temp1, temp2); // area_squared = temp1 * temp2

                                        mpz_sub(temp1, s_p, c); // temp1 = s_p - c
                                        mpz_mul(area_squared, area_squared, temp1); // area_squared *= temp1
                                        mpz_mul(area_squared, area_squared, s_p);   // area_squared *= s_p

                                        if (mpz_cmp_ui(area_squared, 0) > 0) {
                                            mpz_sqrt(area, area_squared);

                                            // Check area-to-perimeter ratio
                                            if (mpz_divisible_p(area, perimeter)) {
                                                mpz_t ratio;
                                                mpz_init(ratio);
                                                mpz_divexact(ratio, area, perimeter);

                                                if (mpz_cmp_ui(ratio, max_ratio) <= 0) {
                                                    // Add to total sum (protected by mutex)
                                                    std::lock_guard<std::mutex> lock(sum_mutex);
                                                    mpz_add(total_sum, total_sum, perimeter);
                                                }

                                                mpz_clear(ratio);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    mpz_clears(a, b, c, perimeter, s_p, area_squared, area, temp1, temp2, NULL);
}

int main() {
    const int max_ratio = 1000;
    const int num_threads = std::thread::hardware_concurrency();
    vector<thread> threads;

    mpz_init(total_sum);
    mpz_set_ui(total_sum, 0);  // Initialize total_sum to 0

    // Start timing
    auto start_time = high_resolution_clock::now();

    // Divide the work dynamically between threads
    int range_per_thread = 18 / num_threads;  // Assuming range for 'p' is from 1 to 18
    for (int i = 0; i < num_threads; ++i) {
        int start = i * range_per_thread + 1;
        int end = (i == num_threads - 1) ? 18 : (i + 1) * range_per_thread;
        threads.emplace_back(compute_heronian, max_ratio, start, end);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    // End timing
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);

    // Output total sum
    cout << "Total sum of perimeters: ";
    mpz_out_str(stdout, 10, total_sum);
    cout << endl;

    // Output time taken
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    // Clear GMP variables
    mpz_clear(total_sum);

    return 0;
}