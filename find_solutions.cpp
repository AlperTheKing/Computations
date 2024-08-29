#include <iostream>
#include <cmath>
#include <omp.h>
#include <vector>

// Function to calculate factorial
unsigned long long factorial(int n) {
    unsigned long long fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

// Function to find the number of solutions
int find_solutions(int max_a, int max_c) {
    int count = 0;

    #pragma omp parallel for reduction(+:count) schedule(dynamic)
    for (int a = 1; a <= max_a; ++a) {
        unsigned long long a_fact = factorial(a);

        for (int c = 1; c <= max_c; ++c) {
            unsigned long long seven_power_c = pow(7, c);
            if (seven_power_c <= a_fact) continue;

            unsigned long long diff = seven_power_c - a_fact;

            // Check if diff is a power of 5
            unsigned long long temp = diff;
            int b = 0;
            while (temp > 1 && temp % 5 == 0) {
                temp /= 5;
                b++;
            }
            if (temp == 1) {
                #pragma omp critical
                {
                    std::cout << "a=" << a << ", b=" << b << ", c=" << c << std::endl;
                }
                count++;
            }
        }
    }

    return count;
}

int main() {
    int max_a = 20;  // Limit for a
    int max_c = 100; // Limit for c

    int num_solutions = find_solutions(max_a, max_c);

    std::cout << "Number of solutions: " << num_solutions << std::endl;

    return 0;
}