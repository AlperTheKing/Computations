#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <string>

// Structure to store solutions and N for sorting later
struct Solution {
    __int128 N;
    int a;
    int b;
    std::string factorization;
};

// Function to check if a number is prime
bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= std::sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

// Generate primes of the form 4k + 1
std::vector<int> generate_primes(int limit) {
    std::vector<int> primes;
    for (int i = 5; i <= limit; i += 4) {
        if (is_prime(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Generate all square-free N values by taking combinations of primes
void generate_square_free_N(const std::vector<int>& primes, std::vector<__int128>& square_free_N) {
    size_t num_primes = primes.size();
    // Use combinations of primes (powerset), except the empty set
    for (size_t i = 1; i < (1 << num_primes); ++i) {
        __int128 N = 1;
        std::string factorization = "";
        for (size_t j = 0; j < num_primes; ++j) {
            if (i & (1 << j)) {
                N *= primes[j];
                if (!factorization.empty()) factorization += "*";
                factorization += std::to_string(primes[j]);
            }
        }
        square_free_N.push_back(N);
    }
}

// Newton's method to compute the integer square root of n for __int128
__int128 integer_sqrt(__int128 n) {
    if (n == 0) return 0;
    __int128 x = n;
    __int128 y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// Helper function to print __int128 values
std::string int128_to_string(__int128 value) {
    if (value == 0) return "0";
    std::string result;
    bool negative = value < 0;
    if (negative) value = -value;
    while (value > 0) {
        result.insert(result.begin(), '0' + (value % 10));
        value /= 10;
    }
    if (negative) result.insert(result.begin(), '-');
    return result;
}

// Find all solutions to a^2 + b^2 = N and store them in a vector for later sorting
void find_solutions(__int128 N, const std::vector<int>& primes, std::vector<Solution>& solutions, __int128 &sum_of_a) {
    for (int a = 0; static_cast<__int128>(a) * a <= N; ++a) {
        __int128 b_squared = N - static_cast<__int128>(a) * a;
        __int128 b = integer_sqrt(b_squared);
        if (b * b == b_squared && a <= b) {
            std::string factorization = int128_to_string(N);  // Record N's factorization
            #pragma omp critical
            {
                solutions.push_back({N, a, static_cast<int>(b), factorization});
            }
            sum_of_a += a;  // Accumulate the sum of 'a' values
        }
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Limit for primes of the form 4k + 1
    int prime_limit = 150;

    // Generate primes of the form 4k + 1
    std::vector<int> primes = generate_primes(prime_limit);

    // Output file
    std::ofstream file("SumofSquares.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file SumofSquares.txt for writing." << std::endl;
        return 1;
    }

    // Generate all square-free N values using combinations of primes
    std::vector<__int128> square_free_N;
    generate_square_free_N(primes, square_free_N);

    std::cout << "Generated " << square_free_N.size() << " square-free N values." << std::endl;

    // Store solutions for sorting later
    std::vector<Solution> solutions;
    __int128 total_sum_of_a = 0;  // To track the total sum of all 'a' values

    // Parallelize the computation to find solutions for each N
    #pragma omp parallel for reduction(+:total_sum_of_a)
    for (size_t i = 0; i < square_free_N.size(); ++i) {
        __int128 N = square_free_N[i];
        find_solutions(N, primes, solutions, total_sum_of_a);
    }

    // Sort the solutions by N in ascending order
    std::sort(solutions.begin(), solutions.end(), [](const Solution& s1, const Solution& s2) {
        return s1.N < s2.N;
    });

    // Write sorted solutions to the file
    for (const auto& sol : solutions) {
        file << "N = " << int128_to_string(sol.N) << " (" << sol.factorization << "), a = " << sol.a << ", b = " << sol.b << std::endl;
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Write total sum of 'a' values and elapsed time to the file
    file << "Total sum of a values: " << int128_to_string(total_sum_of_a) << std::endl;
    file << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // Close the file
    file.close();

    // Print the total sum of 'a' values to the console (since this is the main goal)
    std::cout << "Total sum of a values: " << int128_to_string(total_sum_of_a) << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}