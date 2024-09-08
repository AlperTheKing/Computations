#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include "Random123/philox.h"
#include <cmath>

// Function to calculate the number of digits in an unsigned long long number
int numberOfDigits(unsigned long long n) {
    return std::to_string(n).length();  // Get the length of the number as a string
}

// Function to compute the rounded square root using Heron's method (adapted for integers)
int roundedSquareRoot(unsigned long long n) {
    int digits = numberOfDigits(n);
    unsigned long long x_k;

    // Initial guess based on the number of digits
    if (digits % 2 == 1) {
        x_k = 2 * pow(10, (digits - 1) / 2);
    } else {
        x_k = 7 * pow(10, (digits - 2) / 2);
    }

    unsigned long long x_k1 = 0;
    int iterations = 0;

    while (true) {
        x_k1 = (x_k + (n + x_k - 1) / x_k) / 2;  // The integer division equivalent of the formula

        iterations++;
        if (x_k == x_k1) break;  // If x_k == x_k1, stop the iteration
        x_k = x_k1;
    }

    return iterations;
}

// Function to generate a random 14-digit number using Random123
unsigned long long generateRandomNumber(r123::Philox4x32::ctr_type& ctr, r123::Philox4x32::key_type& key) {
    r123::Philox4x32 rng;
    ctr = rng(ctr, key);
    
    // Generate two parts to create a 14-digit random number
    unsigned long long part1 = ctr[0] % 100000000ULL;  // First 8 digits
    unsigned long long part2 = ctr[1] % 10000000ULL;   // Next 7 digits

    // Combine the two parts to create a 14-digit number
    return part1 * 10000000ULL + part2;
}

int main() {
    const unsigned long long numberOfTests = 10000000000;  // 10 billion tests
    long long sumIterations = 0;

    omp_set_num_threads(64);  // Use all cores on AMD Threadripper 7980X

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop with OpenMP
    #pragma omp parallel reduction(+:sumIterations)
    {
        r123::Philox4x32::ctr_type ctr = {0, 0, 0, 0};
        r123::Philox4x32::key_type key = {52769};  // You can seed this key differently
        ctr[0] = omp_get_thread_num();  // Ensure each thread has a different starting counter

        #pragma omp for schedule(dynamic, 10000)
        for (unsigned long long i = 0; i < numberOfTests; ++i) {
            ctr[1] = i;  // Change the counter for each iteration
            unsigned long long randomNumber = generateRandomNumber(ctr, key);  // Generate random 14-digit number
            int iterations = roundedSquareRoot(randomNumber);  // Find the rounded square root
            sumIterations += iterations;
        }
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