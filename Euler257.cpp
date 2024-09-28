#include <iostream>
<<<<<<< HEAD
#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <pthread.h>

// Function to check if (a+b)(a+c) / (bc) is an integer using int64_t with modulo check
bool is_integer_ratio_mod(int a, int b, int c) {
    int64_t numerator = static_cast<int64_t>(a + b) * (a + c);
    int64_t denominator = static_cast<int64_t>(b) * c;
    return (numerator % denominator == 0);
}

// Helper function for gcd calculation
int64_t gcd(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Pollard's Rho function for factorization
int64_t pollards_rho(int64_t n) {
    if (n % 2 == 0) return 2;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(1, n - 1);

    int64_t x = dist(gen);
    int64_t y = x;
    int64_t c = dist(gen);
    int64_t d = 1;

    while (d == 1) {
        x = (x * x + c) % n;
        y = (y * y + c) % n;
        y = (y * y + c) % n;
        d = gcd(abs(x - y), n);
    }

    return d;
}

// Function to check if (a+b)(a+c) / (bc) is an integer using prime factorization with Pollard's Rho
bool is_integer_ratio_pollards_rho(int a, int b, int c) {
    int64_t numerator = static_cast<int64_t>(a + b) * (a + c);
    int64_t denominator = static_cast<int64_t>(b) * c;
    int64_t gcd_value = gcd(numerator, denominator);
    numerator /= gcd_value;
    denominator /= gcd_value;

    if (denominator == 1) {
        return true;
    }

    while (denominator > 1) {
        int64_t factor = pollards_rho(denominator);
        if (numerator % factor != 0) {
            return false;
        }
        numerator /= factor;
        denominator /= factor;
    }
    return true;
}

// Structure to pass data to threads
struct ThreadData {
    int thread_id;
    int num_threads;
    long long local_count;
    int perimeter_limit;
};

// Thread function for processing triangles
void* process_triangles(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int thread_id = data->thread_id;
    int num_threads = data->num_threads;
    int perimeter_limit = data->perimeter_limit;

    long long local_count = 0;

    for (int a = thread_id + 1; a <= perimeter_limit; a += num_threads) {
        for (int b = a; b + a <= perimeter_limit; ++b) {
            for (int c = b; a + b + c <= perimeter_limit; ++c) {
                if (a == b && b == c) continue;  // Skip equilateral triangles

                if (a + b > c && a + c > b && b + c > a) {
                    bool valid;
                    if (perimeter_limit <= 100000) {
                        valid = is_integer_ratio_mod(a, b, c);  // Use mod method for perimeter <= 100,000
                    } else {
                        valid = is_integer_ratio_pollards_rho(a, b, c);  // Use Pollard's Rho for larger perimeter
                    }
                    if (valid) {
                        local_count++;
                    }
                }
=======
#include <vector>
#include <pthread.h>
#include <mutex>
#include <cmath>
#include <chrono>
#include <atomic>
#include <unistd.h> // For sysconf to get hardware concurrency

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
>>>>>>> aeb3278 (Solution)
            }
        }
    }

<<<<<<< HEAD
    data->local_count = local_count;
    return nullptr;
}

int main() {
    int perimeter_limit;
    std::cout << "Enter the perimeter value: ";
    std::cin >> perimeter_limit;

    // Start timing the execution
    auto start = std::chrono::high_resolution_clock::now();

    long long equilateral_count = perimeter_limit / 3;

    int num_threads = std::thread::hardware_concurrency();
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Launch threads with interleaved work assignment
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].perimeter_limit = perimeter_limit;
        pthread_create(&threads[i], nullptr, process_triangles, (void*)&thread_data[i]);
    }

    long long total_count = equilateral_count;

    // Join threads and accumulate results
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        total_count += thread_data[i].local_count;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total count of valid triangles: " << total_count << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
=======
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
>>>>>>> aeb3278 (Solution)

    return 0;
}