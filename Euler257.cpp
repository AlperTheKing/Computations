#include <iostream>
#include <vector>
#include <pthread.h>
#include <mutex>
#include <cmath>
#include <chrono>
#include <atomic>
#include <unistd.h> // For sysconf to get hardware concurrency
#include <thread>

// Mutex for thread synchronization
std::mutex mtx;

struct Triangle {
    long long a, b, c;
};

// Shared variables
std::vector<Triangle> validTriangles;
long long MAX_PERIMETER = 100000000; // Default value, will be updated by user input

// Efficient integer square root using Newton's method
long long integerSqrt(long long n) {
    if (n == 0 || n == 1)
        return n;
    long long x = n;
    long long y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

struct ThreadData {
    int k;
    int thread_id;
    int total_threads;
};

void* findTriangles(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int k = data->k;
    int thread_id = data->thread_id;
    int total_threads = data->total_threads;

    if (k == 4) {
        // Handle equilateral triangles separately
        for (long long a = thread_id + 1; 3 * a <= MAX_PERIMETER; a += total_threads) {
            Triangle triangle = {a, a, a};
            std::lock_guard<std::mutex> lock(mtx);
            validTriangles.push_back(triangle);
        }
    } else {
        long long m = 2 + 4 * (k - 1); // m = 6 for k=2, m=10 for k=3

        for (long long b = thread_id + 1; b <= MAX_PERIMETER / 2; b += total_threads) {
            for (long long c = b; c <= MAX_PERIMETER - b - 1; ++c) {
                // Compute discriminant D
                __int128 D = (__int128)b * b + (__int128)c * c + (__int128)m * b * c;
                long long s = integerSqrt(D);
                if ((long long)s * s != D) continue; // Not a perfect square
                // Compute 'a'
                long long a_numerator = - (b + c) + s;
                if (a_numerator <= 0 || a_numerator % 2 != 0) continue;
                long long a = a_numerator / 2;
                if (a > b) continue; // Ensure a <= b
                // Check triangle inequalities
                if (a + b <= c || a + c <= b || b + c <= a) continue;
                // Check perimeter constraint
                if (a + b + c > MAX_PERIMETER) continue;
                // Verify that the ratio equals 'k'
                long long ratio_numerator = (a + b) * (a + c);
                long long ratio_denominator = b * c;
                if (ratio_numerator != k * ratio_denominator) continue;
                // Store the valid triangle
                Triangle triangle = {a, b, c};
                std::lock_guard<std::mutex> lock(mtx);
                validTriangles.push_back(triangle);
            }
        }
    }

    pthread_exit(nullptr);
}

int main() {
    // Prompt the user to input the maximum perimeter
    std::cout << "Enter the maximum perimeter: ";
    std::cin >> MAX_PERIMETER;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the number of hardware threads available
    unsigned int total_threads = std::thread::hardware_concurrency();
    if (total_threads == 0) {
        total_threads = sysconf(_SC_NPROCESSORS_ONLN);
        if (total_threads == 0) {
            total_threads = 4; // Default to 4 threads if unable to detect
        }
    }

    std::cout << "Using " << total_threads << " threads.\n";

    // For k = 2, 3, 4
    std::vector<pthread_t> threads(total_threads * 3);
    std::vector<ThreadData> thread_data(threads.size());

    int thread_index = 0;
    for (int k = 2; k <= 4; ++k) {
        for (unsigned int t = 0; t < total_threads; ++t) {
            thread_data[thread_index].k = k;
            thread_data[thread_index].thread_id = t;
            thread_data[thread_index].total_threads = total_threads;
            pthread_create(&threads[thread_index], nullptr, findTriangles, (void*)&thread_data[thread_index]);
            thread_index++;
        }
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        pthread_join(thread, nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the results
    std::cout << "Found " << validTriangles.size() << " valid triangles.\n";
    std::cout << "Time taken: " << elapsed.count() << " seconds.\n";

    // Optionally, print the triangles
    /*
    for (const auto& triangle : validTriangles) {
        std::cout << "a = " << triangle.a << ", b = " << triangle.b << ", c = " << triangle.c << "\n";
    }
    */

    return 0;
}