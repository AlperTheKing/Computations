#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>

using namespace boost::multiprecision;
using namespace std;
using namespace std::chrono;

using boost_int = cpp_int;

std::mutex sum_mutex;
boost_int total_sum = 0;

void compute_heronian(int max_ratio, int start, int end) {
    for (int p = start; p <= end; ++p) {
        for (int w1 = 1; w1 <= 20; ++w1) {
            for (int s = 1; s <= 20; ++s) {
                for (int t = 1; t <= 20; ++t) {
                    for (int u = 1; u <= 20; ++u) {
                        for (int v = 1; v <= 20; ++v) {
                            for (int alpha = 1; alpha <= 20; ++alpha) {
                                for (int beta = 1; beta <= 20; ++beta) {
                                    for (int gamma = 1; gamma <= 20; ++gamma) {
                                        // Calculate the sides of the triangle
                                        boost_int a = p * alpha * u * ((beta * w1 * v) * (beta * w1 * v) + (gamma * s * t) * (gamma * s * t));
                                        boost_int b = p * beta * s * ((alpha * w1 * t) * (alpha * w1 * t) + (gamma * u * v) * (gamma * u * v));
                                        boost_int c = p * (beta * u * v * v + alpha * s * t * t) * (beta * alpha * w1 * w1 - gamma * gamma * s * u);

                                        boost_int perimeter = a + b + c;

                                        // Calculate the semi-perimeter for Heron's formula
                                        boost_int s_p = perimeter / 2;

                                        // Calculate the area using Heron's formula
                                        boost_int area_squared = s_p * (s_p - a) * (s_p - b) * (s_p - c);

                                        if (area_squared > 0) {
                                            boost_int area = sqrt(area_squared);

                                            // Check the area-to-perimeter ratio
                                            if (area % perimeter == 0) {
                                                boost_int ratio = area / perimeter;
                                                if (ratio <= max_ratio) {
                                                    // Lock and update total sum if the ratio condition is met
                                                    std::lock_guard<std::mutex> lock(sum_mutex);
                                                    total_sum += perimeter;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int max_ratio = 1000;
    const int num_threads = std::thread::hardware_concurrency();
    vector<thread> threads;

    // Start timing
    auto start_time = high_resolution_clock::now();

    // Divide the work dynamically between threads
    int range_per_thread = 20 / num_threads;  // Assuming range for 'p' is from 1 to 20
    for (int i = 0; i < num_threads; ++i) {
        int start = i * range_per_thread + 1;
        int end = (i == num_threads - 1) ? 20 : (i + 1) * range_per_thread;
        threads.emplace_back(compute_heronian, max_ratio, start, end);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    // End timing
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);

    cout << "Total sum of perimeters: " << total_sum << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    return 0;
}