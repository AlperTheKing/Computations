#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>

// Function to check if a number is divisible by 3
bool isDivisibleBy3(int num) {
    int sum = 0;
    while (num > 0) {
        sum += num % 10;
        num /= 10;
    }
    return sum % 3 == 0;
}

// Function to check if a number is divisible by 5
bool isDivisibleBy5(int num) {
    int lastDigit = num % 10;
    return lastDigit == 0 || lastDigit == 5;
}

// Function to check if a number is divisible by 7
bool isDivisibleBy7(int num) {
    int lastDigit = num % 10;
    num /= 10;
    num -= 2 * lastDigit;
    return num == 0 || num % 7 == 0;
}

// Function to check if a number is prime using divisibility rules and trial division up to sqrt(number)
bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2 || num == 3 || num == 5 || num == 7) return true;
    if (num % 2 == 0) return false;  // Skip even numbers
    if (isDivisibleBy3(num)) return false;
    if (isDivisibleBy5(num)) return false;
    if (isDivisibleBy7(num)) return false;

    // Trial division for remaining candidates up to sqrt(num)
    int limit = static_cast<int>(sqrt(num));
    for (int i = 11; i <= limit; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

int main() {
    int limit = 1000000000;  // Find primes up to this number
    int primeCount = 0;
    std::vector<int> primes;

    auto start = std::chrono::high_resolution_clock::now();

    // Parallelize the prime-finding loop
    #pragma omp parallel for reduction(+:primeCount) schedule(dynamic)
    for (int num = 2; num <= limit; ++num) {
        if (isPrime(num)) {
            #pragma omp critical
            primes.push_back(num);
            primeCount++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print the first 10 primes
    std::cout << "First 10 primes: ";
    for (int i = 0; i < 10 && i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    // Print the last 10 primes
    std::cout << "Last 10 primes: ";
    for (int i = primes.size() - 10; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Total primes found: " << primeCount << std::endl;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}