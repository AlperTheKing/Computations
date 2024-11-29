#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/sqrt.hpp> // Boost.Math's sqrt
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <iomanip>

using namespace boost::multiprecision;
typedef number<cpp_dec_float<100>> cpp_dec_float_100;

// Mutex for synchronizing access to shared variables
std::mutex mtx;

// Structure to hold the best result
struct Result {
    cpp_dec_float_100 best_xyz;
    cpp_dec_float_100 min_diff;

    Result() : best_xyz(0), min_diff(std::numeric_limits<cpp_dec_float_100>::infinity()) {}
};

// Function to perform the grid search in a given range of x
void grid_search(cpp_dec_float_100 x_start, cpp_dec_float_100 x_end, cpp_dec_float_100 step,
                Result &global_result) {
    cpp_dec_float_100 best_xyz = 0;
    cpp_dec_float_100 min_diff = std::numeric_limits<cpp_dec_float_100>::infinity();

    cpp_dec_float_100 x = x_start;
    while (x <= x_end) {
        cpp_dec_float_100 y = 0.001;
        while (y <= 5.0) { // y must be <=5
            cpp_dec_float_100 z = 9.0 - x - y;
            if (z <= 0 || z > 6.0) {
                y += step;
                continue; // z must be >0 and <=6
            }

            try {
                // Calculate the radicals from Equation 1 using boost::math::sqrt
                cpp_dec_float_100 radical1 = boost::math::sqrt(cpp_dec_float_100(16.0) - x * x);
                cpp_dec_float_100 radical2 = boost::math::sqrt(cpp_dec_float_100(25.0) - y * y);
                cpp_dec_float_100 radical3 = boost::math::sqrt(cpp_dec_float_100(36.0) - z * z);

                // Calculate the sum of radicals
                cpp_dec_float_100 sum_radicals = radical1 + radical2 + radical3;

                // Calculate the difference from 12
                cpp_dec_float_100 diff = abs(sum_radicals - cpp_dec_float_100(12.0));

                // Update local best_xyz and min_diff if this is the closest so far
                if (diff < min_diff) {
                    min_diff = diff;
                    best_xyz = x * y * z;

                    // Early exit if exact solution is found within a small tolerance
                    if (diff < cpp_dec_float_100("0.0001")) {
                        std::lock_guard<std::mutex> lock(mtx);
                        if (diff < global_result.min_diff) {
                            global_result.min_diff = diff;
                            global_result.best_xyz = best_xyz;
                            std::cout << "Exact or near-exact solution found:\n";
                            std::cout << "x = " << x << "\n";
                            std::cout << "y = " << y << "\n";
                            std::cout << "z = " << z << "\n";
                            std::cout << "xyz = " << best_xyz << "\n\n";
                        }
                        return;
                    }
                }
            } catch (...) {
                // In case of any math errors (like negative square roots), skip to next iteration
            }

            // Increment y
            y += step;
        }

        // Increment x
        x += step;
    }

    // Update the global best result if a better one is found in this thread
    std::lock_guard<std::mutex> lock(mtx);
    if (min_diff < global_result.min_diff) {
        global_result.min_diff = min_diff;
        global_result.best_xyz = best_xyz;
    }
}

int main() {
    // Define step size for grid search
    cpp_dec_float_100 step = 0.001;

    // Define the range for x
    cpp_dec_float_100 x_min = 0.001;
    cpp_dec_float_100 x_max = 4.0;

    // Number of threads to use (adjust as per your CPU cores)
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 threads if unable to detect

    std::cout << "Using " << num_threads << " threads for computation.\n";

    // Calculate the range of x for each thread
    cpp_dec_float_100 total_range = x_max - x_min;
    cpp_dec_float_100 range_per_thread = total_range / num_threads;

    // Initialize the global result
    Result global_result;

    // Vector to hold all threads
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        cpp_dec_float_100 start = x_min + i * range_per_thread;
        cpp_dec_float_100 end = (i == num_threads - 1) ? x_max : start + range_per_thread;

        threads.emplace_back(grid_search, start, end, step, std::ref(global_result));
    }

    // Wait for all threads to finish
    for (auto &th : threads) {
        th.join();
    }

    // Display the final approximation
    std::cout << "\nFinal Approximation:\n";
    std::cout << "xyz ≈ " << std::setprecision(100) << global_result.best_xyz << "\n";
    std::cout << "Minimum difference from desired sum (12): " << global_result.min_diff << "\n";

    return 0;
}