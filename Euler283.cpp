#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <numeric>
#include <chrono>
#include <boost/multiprecision/cpp_int.hpp>

namespace mp = boost::multiprecision;

std::mutex mtx;           // Mutex for updating total perimeter sum
std::mutex set_mtx;       // Mutex for updating the set of unique triangles
const int MAX_RATIO = 1000; // The maximum allowed area-to-perimeter ratio
mp::cpp_int total_perimeter_sum = 0; // Global total perimeter sum

// Set to store unique triangles
std::set<std::tuple<mp::cpp_int, mp::cpp_int, mp::cpp_int>> unique_triangles;

// Function to compute GCD of two integers
int gcd(int a, int b) {
    return std::gcd(a, b);
}

// Function to calculate Heronian triangles using Euler's parametrization
void euler_heronian(int start_m, int end_m) {
    // Thread-local storage for unique triangles and local perimeter sum
    std::map<std::tuple<mp::cpp_int, mp::cpp_int, mp::cpp_int>, mp::cpp_int> local_triangles;

    for (int m = start_m; m <= end_m; ++m) {
        for (int n = 1; n < m; ++n) {
            if (gcd(m, n) != 1) continue; // Ensure gcd(m, n) = 1

            for (int p = 1; p <= m; ++p) {
                for (int q = 1; q < p; ++q) {
                    if (gcd(p, q) != 1) continue; // Ensure gcd(p, q) = 1

                    // Apply Euler's parametrization formulas
                    mp::cpp_int a = m * n * (p * p + q * q);
                    mp::cpp_int b = p * q * (m * m + n * n);
                    mp::cpp_int c = (m * q + n * p) * (m * p - n * q);

                    // Ensure sides are positive
                    if (a <= 0 || b <= 0 || c <= 0) continue;

                    mp::cpp_int perimeter = a + b + c;
                    mp::cpp_int area = m * n * p * q * (m * q + n * p) * (m * p - n * q);

                    // Skip invalid cases
                    if (area <= 0 || perimeter <= 0) continue;

                    // Ensure the area is exactly divisible by the perimeter
                    if (area % perimeter != 0) continue;

                    mp::cpp_int ratio = area / perimeter;
                    if (ratio > MAX_RATIO) continue;

                    // Normalize the sides
                    std::vector<mp::cpp_int> sides = {a, b, c};
                    std::sort(sides.begin(), sides.end());
                    auto triangle = std::make_tuple(sides[0], sides[1], sides[2]);

                    // Store in local map
                    local_triangles[triangle] = perimeter;
                }
            }
        }
    }

    // Merge local results into global ones
    {
        std::lock_guard<std::mutex> lock(set_mtx);
        for (const auto& [tri, perim] : local_triangles) {
            if (unique_triangles.find(tri) == unique_triangles.end()) {
                unique_triangles.insert(tri);
                total_perimeter_sum += perim;
            }
        }
    }
}

// Function to perform multithreading with dynamic load balancing
void solve_with_multithreading() {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1; // Fallback to single thread if hardware_concurrency() is not available

    int max_m = 2000; // Estimate a reasonable upper limit for parameter m based on MAX_RATIO

    std::vector<std::thread> threads;
    int range = max_m / num_threads; // Divide work into ranges for each thread

    for (int i = 0; i < num_threads; ++i) {
        int start_m = i * range + 1;
        int end_m = (i == num_threads - 1) ? max_m : (i + 1) * range;

        threads.emplace_back(euler_heronian, start_m, end_m);
    }

    for (auto &t : threads) {
        t.join(); // Wait for all threads to finish
    }
}

int main() {
    // Start the clock using standard C++ chrono
    auto start_time = std::chrono::high_resolution_clock::now();

    // Solve the problem using multithreading
    solve_with_multithreading();

    // End the clock and calculate the elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    // Output the result and the time taken
    std::cout << "Total sum of perimeters: " << total_perimeter_sum << std::endl;
    std::cout << "Number of unique triangles: " << unique_triangles.size() << std::endl;
    std::cout << "Time taken: " << duration << " seconds" << std::endl;

    return 0;
}