#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Precompute factorials of digits 0-9
vector<int> factorials = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};

// Function to calculate the sum of factorials of digits of n (f(n))
int f(int n) {
    int sum = 0;
    while (n > 0) {
        sum += factorials[n % 10]; // Add factorial of the last digit
        n /= 10;
    }
    return sum;
}

// Function to calculate the sum of digits of a number (sf(n))
int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Function to find the smallest n such that sf(f(n)) = i
int g(int i) {
    int n = 1;
    while (true) {
        if (sum_of_digits(f(n)) == i) {
            return n;
        }
        n++;
    }
}

// Function to calculate the sum of digits of g(i)
int sg(int i) {
    return sum_of_digits(g(i));
}

int main() {
    int upper_limit = 150;
    int total_sum = 0;

    // Start time measurement
    auto start = high_resolution_clock::now();

    // Use OpenMP for parallelization
    #pragma omp parallel for reduction(+:total_sum)
    for (int i = 1; i <= upper_limit; i++) {
        total_sum += sg(i);
    }

    // End time measurement
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Output the result
    cout << "Sum of sg(i) for 1 <= i <= " << upper_limit << " is: " << total_sum << endl;
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    return 0;
}