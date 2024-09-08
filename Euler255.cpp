#include <iostream>
#include <gmp.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include "Random123/philox.h"  // Include Random123 for random number generation

// Function to calculate the number of digits in a GMP number
int numberOfDigits(const mpz_t n) {
    return mpz_sizeinbase(n, 10); // GMP function to get the number of digits in base 10
}

// Helper function to multiply by powers of 10 using GMP
void mpz_mul_10exp(mpz_t result, const mpz_t input, unsigned long exp) {
    mpz_t pow10;
    mpz_init(pow10);
    mpz_ui_pow_ui(pow10, 10, exp);  // Calculate 10^exp
    mpz_mul(result, input, pow10);  // result = input * 10^exp
    mpz_clear(pow10);
}

// Function to compute the rounded square root using Heron's method with GMP
int roundedSquareRoot(const mpz_t n) {
    int d = numberOfDigits(n);
    mpz_t x_k, x_k1;
    mpz_inits(x_k, x_k1, NULL);

    // Initialize x_0 based on the number of digits
    if (d % 2 == 1) {
        mpz_set_ui(x_k, 2);
        mpz_mul_10exp(x_k, x_k, (d - 1) / 2);
    } else {
        mpz_set_ui(x_k, 7);
        mpz_mul_10exp(x_k, x_k, (d - 2) / 2);
    }

    int iterations = 0;
    while (true) {
        mpz_fdiv_q(x_k1, n, x_k);     // x_k1 = n / x_k
        mpz_add(x_k1, x_k1, x_k);     // x_k1 = x_k1 + x_k
        mpz_fdiv_q_2exp(x_k1, x_k1, 1); // x_k1 = x_k1 / 2

        iterations++;
        if (mpz_cmp(x_k, x_k1) == 0) break; // if x_k == x_k1, stop the iteration
        mpz_set(x_k, x_k1); // x_k = x_k1
    }

    // Clear memory
    mpz_clears(x_k, x_k1, NULL);
    return iterations;
}

// Function to generate a random 14-digit number using Random123
void generateRandomNumber(mpz_t& result, r123::Philox4x32::ctr_type& ctr, r123::Philox4x32::key_type& key) {
    r123::Philox4x32 rng;
    ctr = rng(ctr, key);
    
    mpz_t part1, part2;
    mpz_inits(part1, part2, NULL);

    // Generate two parts to create a 14-digit random number
    mpz_set_ui(part1, ctr[0] % 100000000);  // Generate first 8 digits
    mpz_set_ui(part2, ctr[1] % 10000000);   // Generate next 7 digits
    
    // Combine the two parts to create a 14-digit number
    mpz_mul_10exp(part1, part1, 7); // Shift part1 to the left by 7 digits
    mpz_add(result, part1, part2);

    // Clear memory
    mpz_clears(part1, part2, NULL);
}

int main() {
    const int numberOfTests = 1000000000; // Number of random 14-digit numbers to test
    long long sumIterations = 0;

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop with OpenMP, ensure each thread has its own mpz_t variables
    #pragma omp parallel reduction(+:sumIterations)
    {
        mpz_t n;
        mpz_init(n); // Initialize mpz_t variable for each thread

        r123::Philox4x32::ctr_type ctr = {0, 0, 0, 0};
        r123::Philox4x32::key_type key = {12345};  // You can seed this key differently

        #pragma omp for
        for (int i = 0; i < numberOfTests; ++i) {
            generateRandomNumber(n, ctr, key); // Generate a random 14-digit number
            int iterations = roundedSquareRoot(n);
            sumIterations += iterations;
        }

        mpz_clear(n); // Clear the mpz_t variable for each thread
    }

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate the average number of iterations
    double averageIterations = static_cast<double>(sumIterations) / numberOfTests;

    // Output the results
    std::cout << "Average number of iterations: " << std::fixed << std::setprecision(10) << averageIterations << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}