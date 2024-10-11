#include <iostream>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <mutex>
#include <chrono>
#include <thread>

using namespace std;
using namespace boost::multiprecision;

mutex result_mutex;  // Mutex to protect shared results

// Newton's method to find the square root of a number
cpp_int newtons_method_sqrt(cpp_int n) {
    cpp_int x = n;
    cpp_int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// Newton's method to check if a number is a perfect square
bool is_perfect_square(cpp_int n) {
    cpp_int sqrt_n = newtons_method_sqrt(n);
    return sqrt_n * sqrt_n == n;
}

// Function to compute sum of squares for a given range [start, end]
cpp_int sum_of_squares(cpp_int start, cpp_int end) {
    cpp_int sum = 0;
    for (cpp_int i = start; i <= end; ++i) {
        sum += i * i;
    }
    return sum;
}

// Function to verify if the given m, n, and k satisfy the main formula
bool verify_formula(cpp_int k, cpp_int m, cpp_int n) {
    // Left-hand side: (k-m)^2 + ... + k^2
    cpp_int lhs = sum_of_squares(k - m, k);

    // Right-hand side: (n+1)^2 + ... + (n+m)^2
    cpp_int rhs = sum_of_squares(n + 1, n + m);

    return lhs == rhs;
}

// Function to compute k1 and verify using the main formula
void check_and_store_pivots(cpp_int c, cpp_int d, cpp_int m, cpp_int n, vector<cpp_int>& pivots) {
    // Compute k1 = (c + d) / 2
    if ((c + d) % 2 == 0) {
        cpp_int k1 = (c + d) / 2;

        // Verify k1 according to the formula (k - m)^2 + ... + k^2 = (n + 1)^2 + ... + (n + m)^2
        if (verify_formula(k1, m, n)) {
            lock_guard<mutex> lock(result_mutex);
            pivots.push_back(k1);
            cout << "Verified! m = " << m << ", n = " << n << ", k1 = " << k1 << endl;
        }
    }
}

// Parallelized function to handle n = k case, where k = 2m(m+1)
void handle_n_equals_k_case_parallel(cpp_int max_limit, vector<cpp_int>& pivots, cpp_int start_m, cpp_int end_m) {
    for (cpp_int m = start_m; m <= end_m; ++m) {
        cpp_int k = 2 * m * (m + 1);  // k = 2m(m+1)

        if (k > max_limit) {
            break;  // Stop if k exceeds the maximum limit
        }

        lock_guard<mutex> lock(result_mutex);
        pivots.push_back(k);
        cout << "m = " << m << ", n = " << k << ", k = " << k << endl;
    }
}

// Parallelized function to find n > k pairs using the formula d^2 = c^2 + 4cn
void find_n_greater_than_k_pairs_parallel(cpp_int max_limit, vector<cpp_int>& pivots, cpp_int start_m, cpp_int end_m) {
    for (cpp_int m = start_m; m <= end_m; ++m) {  // m > 0 enforced here
        cpp_int c = m * (m + 1);  // c = m(m+1)

        if (c > max_limit) {
            break;  // Stop if c exceeds the maximum limit
        }

        for (cpp_int n = c + 1;; ++n) {  // Ensure n > k
            cpp_int rhs = c * c + 4 * c * n;  // d^2 = c^2 + 4cn

            // Check if rhs is a perfect square using Newton's method
            if (is_perfect_square(rhs)) {
                cpp_int d = newtons_method_sqrt(rhs);  // Compute d using Newton's method
                check_and_store_pivots(c, d, m, n, pivots);
            }

            // Stop if 4cn exceeds max_limit, refine condition to avoid missed pairs
            if (rhs > max_limit * max_limit) {
                break;
            }
        }
    }
}

// Function to divide tasks across threads and manage multithreading
void parallel_execution(cpp_int max_limit, vector<cpp_int>& pivots, int num_threads) {
    // Determine the range of work for each thread
    cpp_int total_m = max_limit / 2;  // The maximum value of m is approximately max_limit / 2

    cpp_int range_per_thread = total_m / num_threads;

    vector<thread> threads;

    // Launch threads for the n = k case
    for (int i = 0; i < num_threads; ++i) {
        cpp_int start_m = i * range_per_thread + 1;
        cpp_int end_m = (i == num_threads - 1) ? total_m : (i + 1) * range_per_thread;
        threads.push_back(thread(handle_n_equals_k_case_parallel, max_limit, ref(pivots), start_m, end_m));
    }

    // Join threads for n = k case
    for (auto& t : threads) {
        t.join();
    }

    threads.clear();  // Clear threads before launching the next set

    // Launch threads for the n > k case
    for (int i = 0; i < num_threads; ++i) {
        cpp_int start_m = i * range_per_thread + 1;
        cpp_int end_m = (i == num_threads - 1) ? total_m : (i + 1) * range_per_thread;
        threads.push_back(thread(find_n_greater_than_k_pairs_parallel, max_limit, ref(pivots), start_m, end_m));
    }

    // Join threads for n > k case
    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    cpp_int limit = 1000;  // For your example, set limit = 1000
    vector<cpp_int> pivots;

    // Measure execution time
    auto start_time = chrono::high_resolution_clock::now();

    // Number of threads to use for parallelization
    int num_threads = thread::hardware_concurrency();  // Use the available number of threads

    // Execute tasks in parallel
    parallel_execution(limit, pivots, num_threads);

    // Measure the elapsed time
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    // Remove duplicates (since the same pivot can be found across different m or n values)
    sort(pivots.begin(), pivots.end());
    pivots.erase(unique(pivots.begin(), pivots.end()), pivots.end());

    // Calculate the sum of all distinct square-pivots
    cpp_int total_sum = 0;
    for (cpp_int pivot : pivots) {
        total_sum += pivot;
    }

    cout << "Sum of all distinct square-pivots <= " << limit << ": " << total_sum << endl;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}