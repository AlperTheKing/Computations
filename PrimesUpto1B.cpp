#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <algorithm> // Include the algorithm header for std::sort

// Function to check if a number is prime
bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    std::vector<int> primes;
    int count = 0;

    // Start measuring time
    std::clock_t start_time = std::clock();

    // Finding prime numbers in parallel
    #pragma omp parallel for schedule(dynamic) reduction(+:count)
    for (int i = 0; i < 1000000000; ++i) {
        if (is_prime(i)) {
            #pragma omp critical
            primes.push_back(i);
            count++;
        }
    }

    // End time measurement
    std::clock_t end_time = std::clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;

    // Sort the primes vector
    std::sort(primes.begin(), primes.end());

    // Print the results
    std::cout << count << " prime numbers found" << std::endl;
    std::cout << "First 10 prime numbers: ";
    for (int i = 0; i < 10 && i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last 10 prime numbers: ";
    for (int i = primes.size() - 10; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << elapsed_time << " seconds elapsed." << std::endl;

    return 0;
}