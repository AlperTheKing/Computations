#include <iostream>
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
            }
        }
    }

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

    return 0;
}