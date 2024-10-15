#include <iostream>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
#include <mutex>

using namespace std;

// Function to check if a triangle with sides a, b, c is valid
bool isValidTriangle(int64_t a, int64_t b, int64_t c) {
    return (a + b > c && a + c > b && b + c > a);
}

// Function to compute 16 * Area^2 using integer Heron's formula
int64_t compute16AreaSquared(int64_t a, int64_t b, int64_t c) {
    int64_t s = a + b + c;
    int64_t s_minus_2a = s - 2LL * a;
    int64_t s_minus_2b = s - 2LL * b;
    int64_t s_minus_2c = s - 2LL * c;
    return s * s_minus_2a * s_minus_2b * s_minus_2c;
}

// Function to perform computation for a range of k values
void computePerimeterSum(int64_t k_start, int64_t k_end, int64_t& partialSum, mutex& mtx) {
    int64_t localSum = 0;

    for (int64_t k = k_start; k <= k_end; ++k) {
        // Based on observations, we can set bounds for x
        // Let's assume x ranges from k to some multiple of k
        int64_t x_start = k;
        int64_t x_end = 4 * k * k + 1000; // Adjust as needed

        for (int64_t x = x_start; x <= x_end; ++x) {
            for (int64_t y = x; y <= x_end; ++y) {
                // Compute numerator and denominator to solve for z
                int64_t numerator = 16LL * k * k * (x + y);
                int64_t denominator = x * y - 16LL * k * k;

                if (denominator <= 0)
                    continue;

                if (numerator % denominator != 0)
                    continue;

                int64_t z = numerator / denominator;
                if (z < y)
                    continue;

                // Verify the original equation
                if (x * y * z != 16LL * k * k * (x + y + z))
                    continue;

                // Compute perimeter P
                int64_t P = x + y + z;

                // Compute sides a, b, c
                int64_t a_times_2 = P - x;
                int64_t b_times_2 = P - y;
                int64_t c_times_2 = P - z;

                // Check if sides are integers
                if (a_times_2 % 2 != 0 || b_times_2 % 2 != 0 || c_times_2 % 2 != 0)
                    continue;

                int64_t a = a_times_2 / 2;
                int64_t b = b_times_2 / 2;
                int64_t c = c_times_2 / 2;

                // Check if sides are positive
                if (a <= 0 || b <= 0 || c <= 0)
                    continue;

                // Check if sides form a valid triangle
                if (!isValidTriangle(a, b, c))
                    continue;

                // Compute 16 * Area^2
                int64_t areaSquared16 = compute16AreaSquared(a, b, c);

                // Compute 16k^2 * P^2
                int64_t expectedAreaSquared16 = 16LL * k * k * P * P;

                // Verify that area squared matches expected value
                if (areaSquared16 != expectedAreaSquared16)
                    continue;

                // Update the local sum
                localSum += P;

                // Optionally, print the found triangle
                // Comment out to reduce console output and improve performance
                
                mtx.lock();
                std::cout << "Found triangle with sides: " << a << ", " << b << ", " << c << ", k = " << k << std::endl;
                std::cout << "Perimeter: " << P << std::endl;
                mtx.unlock();
                
            }
        }
    }

    // Safely update the shared total sum
    std::lock_guard<std::mutex> lock(mtx);
    partialSum += localSum;
}

int main() {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    const int64_t MAX_K = 1000;
    int64_t totalPerimeterSum = 0;

    // Determine the number of threads to use
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
        numThreads = 4; // Default to 4 threads if hardware_concurrency returns 0

    // Divide the k range among threads
    int64_t k_per_thread = MAX_K / numThreads;
    vector<thread> threads;
    mutex mtx;

    // Launch threads
    int64_t k_current = 1;
    for (unsigned int i = 0; i < numThreads; ++i) {
        int64_t k_start = k_current;
        int64_t k_end = (i == numThreads - 1) ? MAX_K : k_current + k_per_thread - 1;
        k_current = k_end + 1;

        threads.emplace_back(computePerimeterSum, k_start, k_end, std::ref(totalPerimeterSum), std::ref(mtx));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    std::cout << "Total Sum of Perimeters: " << totalPerimeterSum << std::endl;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}