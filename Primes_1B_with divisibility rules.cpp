#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <stdexcept>

// Function to split digits of a number
std::vector<int> split_digits(int n) {
    std::vector<int> digits;
    while (n > 0) {
        digits.insert(digits.begin(), n % 10);
        n /= 10;
    }
    return digits;
}

// Divisibility check functions
bool divisible_by_2(int n) {
    return n % 2 == 0;
}

bool divisible_by_3(int n) {
    std::vector<int> digits = split_digits(n);
    int sum = 0;
    for (int digit : digits) {
        sum += digit;
    }
    return sum % 3 == 0;
}

bool divisible_by_5(int n) {
    int last_digit = n % 10;
    return last_digit == 5 || last_digit == 0;
}

bool divisible_by_7(int n) {
    std::vector<int> digits = split_digits(n);
    if (digits.size() == 1) return false;
    
    int combined_number = 0;
    for (size_t i = 0; i < digits.size() - 1; ++i) {
        combined_number = combined_number * 10 + digits[i];
    }
    int last_digit = digits.back();
    
    while (combined_number >= 10) {
        combined_number -= last_digit * 2;
    }
    return combined_number % 7 == 0;
}

// Find the appropriate multiplier 'm' based on the last digit of D
int find_m(int D) {
    int last_digit = D % 10;
    switch (last_digit) {
        case 1: return 9;
        case 3: return 3;
        case 7: return 7;
        case 9: return 1;
        default: throw std::invalid_argument("D should end in 1, 3, 7, or 9");
    }
}

// Generalized divisibility check
bool generalized_divisibility_check(int N, int D) {
    if (D % 10 != 1 && D % 10 != 3 && D % 10 != 7 && D % 10 != 9) {
        throw std::invalid_argument("D should end in 1, 3, 7, or 9");
    }
    int m = find_m(D);
    
    while (N >= 10) {
        int t = N / 10;  // All digits except the last one
        int q = N % 10;  // Last digit
        N = m * q + t;   // Transform the number
    }
    return N % D == 0;
}

// Check if D is prime without using the is_prime function
bool check_if_D_is_prime(int D) {
    if (D < 2) return false;
    if (divisible_by_2(D) || divisible_by_3(D) || divisible_by_5(D)) return false;
    if (divisible_by_7(D)) return false;
    for (int i = 11; i <= static_cast<int>(std::sqrt(D)); i += 2) {
        if (i % 10 == 1 || i % 10 == 3 || i % 10 == 7 || i % 10 == 9) {
            if (generalized_divisibility_check(D, i)) {
                return false;
            }
        }
    }
    return true;
}

// Function to find primes in a given range
void find_primes_in_range(int start, int end, std::vector<int>& primes) {
    for (int i = start; i < end; ++i) {
        if (divisible_by_2(i)) continue;
        if (divisible_by_3(i)) continue;
        if (divisible_by_5(i)) continue;
        if (divisible_by_7(i)) continue;

        // Generalized divisibility rule for D that is prime
        bool is_prime = true;
        for (int D = 11; D <= static_cast<int>(std::sqrt(i)); D += 2) {
            if (D % 10 == 1 || D % 10 == 3 || D % 10 == 7 || D % 10 == 9) {  // Ensure D ends in 1, 3, 7, or 9
                if (check_if_D_is_prime(D) && generalized_divisibility_check(i, D)) {
                    is_prime = false;
                    break;
                }
            }
        }

        if (is_prime) {
            primes.push_back(i);
        }
    }
}

int main() {
    const int MAX_NUMBER = 1000000;
    const int NUM_THREADS = std::thread::hardware_concurrency(); // Get the number of available cores
    const int RANGE_PER_THREAD = MAX_NUMBER / NUM_THREADS;

    std::vector<int> primes;
    std::vector<std::thread> threads;
    std::vector<std::vector<int>> primes_per_thread(NUM_THREADS);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads to find primes in parallel
    for (int t = 0; t < NUM_THREADS; ++t) {
        int start = t * RANGE_PER_THREAD + 2; // Start range for this thread
        int end = (t == NUM_THREADS - 1) ? MAX_NUMBER : start + RANGE_PER_THREAD;
        threads.emplace_back(find_primes_in_range, start, end, std::ref(primes_per_thread[t]));
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Combine the results from all threads
    for (const auto& thread_primes : primes_per_thread) {
        primes.insert(primes.end(), thread_primes.begin(), thread_primes.end());
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Printing the first 30 and last 30 prime numbers
    std::cout << "First 30 primes:" << std::endl;
    for (size_t i = 0; i < 30 && i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\nLast 30 primes:" << std::endl;
    for (size_t i = primes.size() > 30 ? primes.size() - 30 : 0; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\nNumber of primes found: " << primes.size() << std::endl;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}