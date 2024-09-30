#include <iostream>
#include <vector>
#include <pthread.h>
#include <set>
#include <tuple>
#include <string>
#include <sstream>
#include <chrono>
#include <mutex>
#include <unistd.h> // For sysconf()

typedef unsigned long long ull;
typedef __uint128_t ull128; // Use 128-bit integers for large numbers

std::mutex mtx;
std::set<ull> square_pivots; // Set to store unique square-pivots

// Custom output function for __int128_t
std::ostream& operator<<(std::ostream& dest, __uint128_t value) {
    std::ostream::sentry s(dest);
    if (s) {
        __uint128_t tmp = value;
        char buffer[128];
        char* d = std::end(buffer);
        do {
            --d;
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        dest.write(d, std::end(buffer) - d);
    }
    return dest;
}

// Integer square root calculation for 128-bit integers
ull128 int_sqrt(ull128 x) {
    ull128 left = 0, right = x;
    while (left < right) {
        ull128 mid = (left + right + 1) / 2;
        if (mid * mid <= x)
            left = mid;
        else
            right = mid - 1;
    }
    return left;
}

// Struct for passing parameters to the pthread function
struct ThreadData {
    ull start_k;
    ull end_k;
    ull max_value;
};

// Function to check for valid square-pivot k and print the equations
void* find_square_pivots(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    ull start_k = data->start_k;
    ull end_k = data->end_k;
    ull max_value = data->max_value;

    for (ull k = start_k; k <= end_k; ++k) {
        ull k_squared = k * k;
        for (ull m = 1; m < k; ++m) { // Ensure m > 0
            ull left = k;  // Ensure n >= k
            ull right = max_value;
            while (left <= right) {
                ull n = (left + right) / 2;
                ull lhs = m * (n + k) * (n - k + m + 1);
                if (lhs == k_squared) {
                    // Valid pair found, print equation
                    std::lock_guard<std::mutex> lock(mtx);
                    square_pivots.insert(k);
                    std::cout << "m = " << m << ", n = " << n << ", k = " << k 
                              << " -> " << m << "(" << n << " + " << k << ")(" << n 
                              << " - " << k << " + " << m << " + 1) = " << k << "^2\n";
                    break;
                } else if (lhs < k_squared) {
                    left = n + 1;
                } else {
                    right = n - 1;
                }
            }
        }
    }
    return nullptr;
}

int main() {
    ull max_value;
    std::cout << "Enter the maximum value for k: ";
    std::cin >> max_value;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get the number of CPU cores
    unsigned int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads == 0)
        num_threads = 4; // Default to 4 if sysconf cannot determine

    std::cout << "Number of threads: " << num_threads << std::endl;

    // Split work between threads
    ull values_per_thread = max_value / num_threads;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Launch threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        ull start_k = i * values_per_thread + 1;
        ull end_k = (i == num_threads - 1) ? max_value : (i + 1) * values_per_thread;

        thread_data[i].start_k = start_k;
        thread_data[i].end_k = end_k;
        thread_data[i].max_value = max_value;

        pthread_create(&threads[i], nullptr, find_square_pivots, &thread_data[i]);
    }

    // Wait for all threads to complete
    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Calculate sum of all distinct square-pivots
    ull sum_of_square_pivots = 0;
    for (const auto& k : square_pivots) {
        sum_of_square_pivots += k;
    }

    // Output the result
    std::cout << "Sum of all distinct square-pivots <= " << max_value << ": " << sum_of_square_pivots << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}