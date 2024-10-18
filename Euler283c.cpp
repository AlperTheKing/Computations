#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#include <boost/multiprecision/cpp_int.hpp>
#include <set>
#include <tuple>
#include <algorithm> // For std::sort

using namespace std;
using namespace boost::multiprecision;

// Function to check if a triangle with sides a, b, c is valid
bool isValidTriangle(const cpp_int& a, const cpp_int& b, const cpp_int& c) {
    return (a + b > c && a + c > b && b + c > a);
}

// Function to compute 16 * Area^2 using integer Heron's formula
cpp_int compute16AreaSquared(const cpp_int& a, const cpp_int& b, const cpp_int& c) {
    cpp_int s = a + b + c;
    cpp_int s_minus_2a = s - 2 * a;
    cpp_int s_minus_2b = s - 2 * b;
    cpp_int s_minus_2c = s - 2 * c;
    return s * s_minus_2a * s_minus_2b * s_minus_2c;
}

// Shared set to store unique triangles and its mutex
set<tuple<cpp_int, cpp_int, cpp_int>> triangleSet;
mutex setMutex;

// Worker function for threads
void workerFunction(queue<int64_t>& kQueue, mutex& queueMutex, cpp_int& totalSum, mutex& sumMutex) {
    cpp_int localSum = 0;

    while (true) {
        int64_t k;

        // Lock the queue to safely access it
        {
            lock_guard<mutex> lock(queueMutex);
            if (kQueue.empty())
                break;
            k = kQueue.front();
            kQueue.pop();
        }

        cpp_int k_cpp = k;

        // Perform computation for the current k
        cpp_int x_start = k;
        cpp_int x_end = 4 * k * k + 1000; // Adjust as needed

        for (cpp_int x = x_start; x <= x_end; ++x) {
            for (cpp_int y = x; y <= x_end; ++y) {
                // Compute numerator and denominator to solve for z
                cpp_int numerator = 16 * k_cpp * k_cpp * (x + y);
                cpp_int denominator = x * y - 16 * k_cpp * k_cpp;

                if (denominator <= 0)
                    continue;

                if (numerator % denominator != 0)
                    continue;

                cpp_int z = numerator / denominator;
                if (z < y)
                    continue;

                // Verify the original equation
                if (x * y * z != 16 * k_cpp * k_cpp * (x + y + z))
                    continue;

                // Compute perimeter P
                cpp_int P = x + y + z;

                // Compute sides a, b, c
                cpp_int a_times_2 = P - x;
                cpp_int b_times_2 = P - y;
                cpp_int c_times_2 = P - z;

                // Check if sides are integers
                if (a_times_2 % 2 != 0 || b_times_2 % 2 != 0 || c_times_2 % 2 != 0)
                    continue;

                cpp_int a = a_times_2 / 2;
                cpp_int b = b_times_2 / 2;
                cpp_int c = c_times_2 / 2;

                // Check if sides are positive
                if (a <= 0 || b <= 0 || c <= 0)
                    continue;

                // Check if sides form a valid triangle
                if (!isValidTriangle(a, b, c))
                    continue;

                // Compute 16 * Area^2
                cpp_int areaSquared16 = compute16AreaSquared(a, b, c);

                // Compute 16k^2 * P^2
                cpp_int expectedAreaSquared16 = 16 * k_cpp * k_cpp * P * P;

                // Verify that area squared matches expected value
                if (areaSquared16 != expectedAreaSquared16)
                    continue;

                // Sort the sides to create a unique identifier
                vector<cpp_int> sides = {a, b, c};
                sort(sides.begin(), sides.end());
                tuple<cpp_int, cpp_int, cpp_int> triangle(sides[0], sides[1], sides[2]);

                // Check for uniqueness
                {
                    lock_guard<mutex> lock(setMutex);
                    if (triangleSet.find(triangle) != triangleSet.end())
                        continue; // Triangle already counted
                    triangleSet.insert(triangle);
                }

                // Update the local sum
                localSum += P;

                // Optionally collect results for later output
                // For performance, avoid printing from threads
            }
        }
    }

    // Safely update the shared total sum
    {
        lock_guard<mutex> lock(sumMutex);
        totalSum += localSum;
    }
}

int main() {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    const int64_t MAX_K = 1000;
    cpp_int totalPerimeterSum = 0;

    // Determine the number of threads to use
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
        numThreads = 4; // Default to 4 threads if hardware_concurrency returns 0

    // Create a shared queue of k values
    queue<int64_t> kQueue;
    for (int64_t k = 1; k <= MAX_K; ++k) {
        kQueue.push(k);
    }

    mutex queueMutex;
    mutex sumMutex;
    vector<thread> threads;

    // Launch threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back(workerFunction, ref(kQueue), ref(queueMutex), ref(totalPerimeterSum), ref(sumMutex));
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