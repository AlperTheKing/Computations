#include <iostream>
#include <cmath>
#include <set>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to calculate the sum of squares from a to b
unsigned long long sum_of_squares(int a, int b) {
    unsigned long long sum = 0;
    for (int i = a; i <= b; ++i) {
        sum += i * i;
    }
    return sum;
}

int main() {
    const unsigned long long LIMIT = 10000000000ULL;  // 10^10
    set<unsigned long long> square_pivots;
    
    // Start time measurement
    auto start = high_resolution_clock::now();

    // Parallelized outer loop over k
    #pragma omp parallel for schedule(dynamic)
    for (unsigned long long k = 1; k <= sqrt(LIMIT); ++k) {
        for (int m = 1; m < k; ++m) {
            unsigned long long left_sum = sum_of_squares(k - m, k);  // Sum of m+1 consecutive squares ending at k
            
            // Now try to find n such that the right sum matches the left sum
            int n = k;
            unsigned long long right_sum = 0;
            for (int j = 1; j <= m; ++j) {
                right_sum += (n + j) * (n + j);  // Sum of squares from n+1 to n+m
            }
            
            if (left_sum == right_sum) {
                #pragma omp critical
                square_pivots.insert(k);  // Insert valid k into the set to ensure uniqueness
            }
        }
    }

    // Calculate the sum of all distinct square pivots
    unsigned long long total_sum = 0;
    for (auto k : square_pivots) {
        total_sum += k;
    }

    // Stop time measurement
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Output the results
    cout << "Sum of all distinct square-pivots <= " << LIMIT << " is: " << total_sum << endl;
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    return 0;
}