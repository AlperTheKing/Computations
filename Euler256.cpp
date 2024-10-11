#include <flint/flint.h>
#include <flint/fmpz_poly.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

// Forward declaration of the function
bool is_tatami_free(int a, int b);

std::mutex print_mutex;  // Mutex to protect printing to console
std::mutex count_mutex;  // Mutex to protect access to shared counts

// Shared vector to store the first 10 areas for T(s) = 1, 2, ... 10 and T(s) = 200
std::vector<int> areas(201, -1);  // Index corresponds to T(s)

// Function to calculate the number of tatami-free configurations for a given area s
int count_tatami_free_configurations(int s) {
    int count = 0;

    // Loop over all divisors of s to get a and b such that a * b = s
    for (int a = 1; a * a <= s; ++a) {
        if (s % a == 0) {
            int b = s / a;  // b is the other divisor

            // Ensure that a <= b
            if (a <= b) {
                // For each (a, b) pair, check if it is tatami-free
                if (is_tatami_free(a, b)) {
                    count++;
                }
            }
        }
    }

    return count;
}

// Function to check if a given room size a x b is tatami-free
bool is_tatami_free(int a, int b) {
    // For odd areas, it's impossible to fully tile with 1x2 dominos
    if ((a * b) % 2 != 0) {
        return false;
    }

    // Otherwise, assume the room is tatami-free
    return true;
}

// Function to search for tatami-free configurations and store the first 10 areas and T(s) = 200
void search_tatami_areas(int start_s, int end_s) {
    for (int s = start_s; s <= end_s; s += 2) {  // Check only even s
        int tatami_free_count = count_tatami_free_configurations(s);

        // Lock access to shared data for printing and updating the first 10 areas and T(s) = 200
        std::lock_guard<std::mutex> lock(print_mutex);
        if (tatami_free_count >= 1 && tatami_free_count <= 10 && areas[tatami_free_count] == -1) {
            areas[tatami_free_count] = s;
            std::cout << "T(" << s << ") = " << tatami_free_count << std::endl;
        }
        if (tatami_free_count == 200 && areas[200] == -1) {
            areas[200] = s;
            std::cout << "T(" << s << ") = 200" << std::endl;
        }

        // Break out once all the first 10 values and T(s) = 200 are found
        bool all_found = true;
        for (int i = 1; i <= 10; ++i) {
            if (areas[i] == -1) {
                all_found = false;
                break;
            }
        }
        if (areas[200] == -1) {
            all_found = false;
        }

        if (all_found) break;
    }
}

int main() {
    // Measure start time
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = std::thread::hardware_concurrency();  // Number of available threads
    int max_s = 100000;  // Define a reasonable range of areas to search

    // Divide the area range among threads
    int range_per_thread = max_s / num_threads;

    // Start threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        int start_s = i * range_per_thread + 2;  // Start from the first even number
        int end_s = (i == num_threads - 1) ? max_s : start_s + range_per_thread - 1;
        threads.emplace_back(search_tatami_areas, start_s, end_s);
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    // Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds." << std::endl;

    return 0;
}