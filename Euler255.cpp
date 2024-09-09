#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cmath>

// Function to calculate the number of digits in an unsigned long long number
int numberOfDigits(unsigned long long n) {
    return std::to_string(n).length();
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

int main() {
    const unsigned long long startRange = 10000000000000ULL;  // Start of the range 10^13
    const unsigned long long endRange = 100000000000000ULL;   // End of the range 10^14 (exclusive)
    unsigned long long totalIterations = 0;                   // To accumulate total iterations
    unsigned long long numberOfNumbers = endRange - startRange;  // Total numbers in the range

    omp_set_num_threads(64);  // Use all cores on AMD Threadripper 7980X

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop with OpenMP
    #pragma omp parallel for reduction(+:totalIterations)
    for (unsigned long long n = startRange; n < endRange; ++n) {
        int iterations = roundedSquareRoot(n);  // Calculate iterations for number n
        totalIterations += iterations;          // Accumulate total iterations
    }

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate the average number of iterations
    double averageIterations = static_cast<double>(totalIterations) / numberOfNumbers;

    // Output the results
    std::cout << "Average number of iterations for the range [10^13, 10^14): " 
              << std::fixed << std::setprecision(10) << averageIterations << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}