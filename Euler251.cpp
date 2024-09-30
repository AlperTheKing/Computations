#include <iostream>
#include <vector>
#include <pthread.h>
#include <set>
#include <tuple>
#include <string>
#include <chrono>
#include <mutex>
#include <unistd.h> // For sysconf()
#include <algorithm> // For std::min

typedef unsigned long long ull;
typedef __uint128_t ull128; // Use 128-bit integers for large numbers

std::mutex result_mutex; // Mutex for thread-safe result merging
std::mutex a_mutex; // Mutex for protecting 'current_a'

// Custom input function for __uint128_t
std::istream& operator>>(std::istream& is, __uint128_t& value) {
    std::string str;
    is >> str;
    value = 0;
    for (char c : str) {
        if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
        } else {
            is.setstate(std::ios::failbit); // Set error state if invalid character found
            break;
        }
    }
    return is;
}

// Struct for passing parameters to the pthread function
struct ThreadData {
    ull128 max_sum;
    ull128 chunk_size;
    std::set<std::tuple<ull128, ull128, ull128>>* all_triplets;
    ull128* current_a;
};

// Manual integer square root calculation for 128-bit integers
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

// Function to find Cardano Triplets in a given range of 'a'
void* find_cardano_triplets(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    ull128 max_sum = data->max_sum;
    ull128 chunk_size = data->chunk_size;
    ull128* current_a = data->current_a;

    ull128 start_a;
    while (true) {
        // Use mutex to safely access and update 'current_a'
        {
            std::lock_guard<std::mutex> lock(a_mutex);
            start_a = *current_a;
            *current_a += chunk_size;
        }
        
        if (start_a > max_sum) break;

        ull128 end_a = std::min(start_a + chunk_size - 1, max_sum);
        std::set<std::tuple<ull128, ull128, ull128>> local_triplets;

        for (ull128 a = start_a; a <= end_a; a += 3) { // a ≡ 2 mod 3
            ull128 N = (1 + a) * (1 + a) * (8 * a - 1);
            if (N % 27 != 0) continue;
            ull128 N_div = N / 27;

            for (ull128 b = 1; b * b <= N_div; ++b) {
                if (N_div % (b * b) == 0) {
                    ull128 c = N_div / (b * b);
                    if (a + b + c <= max_sum) {
                        // Ensure b <= c to avoid duplicates
                        if (b > c) std::swap(b, c);
                        local_triplets.insert(std::make_tuple(a, b, c));
                    }
                }
            }
        }

        // Merge local results into the global result set using mutex
        std::lock_guard<std::mutex> lock(result_mutex);
        data->all_triplets->insert(local_triplets.begin(), local_triplets.end());
    }

    return nullptr;
}

int main() {
    ull128 max_sum;
    std::cout << "Enter the maximum value for (a + b + c): ";
    std::cin >> max_sum;  // Custom input handling for __uint128_t

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get the number of CPU cores
    unsigned int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads == 0) num_threads = 4; // Default to 4 if sysconf cannot determine

    std::cout << "Number of threads: " << num_threads << std::endl;

    // Prepare for dynamic scheduling with thread-local results
    std::vector<pthread_t> threads(num_threads);
    std::set<std::tuple<ull128, ull128, ull128>> all_triplets;
    ull128 current_a = 2; // Start from a = 2

    ull128 chunk_size = 1000; // Each thread processes chunks of 1000 `a` values at a time

    ThreadData thread_data{max_sum, chunk_size, &all_triplets, &current_a};

    // Launch threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], nullptr, find_cardano_triplets, &thread_data);
    }

    // Wait for all threads to complete
    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Output the total number of unique Cardano triplets found
    std::cout << "Total Cardano Triplets: " << all_triplets.size() << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}