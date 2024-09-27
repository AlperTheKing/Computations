#include <iostream>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <thread>  // For hardware_concurrency()

#include "random123/xoshiro256plusplus.h"  // Include the xoshiro256plusplus.h

#define NUM_SIMULATIONS 1000000000  // Define total number of simulations

using namespace std;
using namespace std::chrono;

// Coin flipping results
const string pattern1 = "HTTTH";
const string pattern2 = "HTTHH";

// Thread data structure
struct ThreadData {
    int id;
    int count1;
    int count2;
    int num_simulations;
};

// Function to simulate coin flips and look for patterns
void* simulate_flips(void* threadarg) {
    ThreadData* data = (ThreadData*)threadarg;

    string flips;
    int count1 = 0, count2 = 0;

    for (int i = 0; i < data->num_simulations; i++) {
        flips.clear();

        while (true) {
            // Generate random bit (0 for T, 1 for H)
            uint64_t rand_value = next();  // Use the next() function from xoshiro256plusplus.h
            int coin_flip = rand_value % 2;  // either 0 (Tails) or 1 (Heads)

            flips += (coin_flip == 0) ? 'T' : 'H';
            if (flips.size() > 5) {
                flips.erase(0, 1);  // Keep only the last 5 flips
            }

            // Check for pattern
            if (flips == pattern1) {
                count1++;
                break;
            } else if (flips == pattern2) {
                count2++;
                break;
            }
        }
    }

    data->count1 = count1;
    data->count2 = count2;

    pthread_exit(NULL);
}

int main() {
    // Initialize the state (s) for xoshiro256++
    s[0] = 123456789;
    s[1] = 987654321;
    s[2] = 111111111;
    s[3] = 222222222;

    // Get the number of hardware threads available on the system
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;  // Default to 4 threads if hardware_concurrency() fails
    }

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int total_count1 = 0, total_count2 = 0;

    auto start = high_resolution_clock::now();

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].id = i;
        thread_data[i].count1 = 0;
        thread_data[i].count2 = 0;
        thread_data[i].num_simulations = NUM_SIMULATIONS / num_threads;
        int rc = pthread_create(&threads[i], NULL, simulate_flips, (void*)&thread_data[i]);
        if (rc) {
            cout << "Error: Unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    // Join threads and collect results
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_count1 += thread_data[i].count1;
        total_count2 += thread_data[i].count2;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Output results
    cout << "Pattern " << pattern1 << " appeared: " << total_count1 << " times\n";
    cout << "Pattern " << pattern2 << " appeared: " << total_count2 << " times\n";
    cout << "Total simulations: " << NUM_SIMULATIONS << "\n";
    cout << "Time taken: " << duration.count() << " milliseconds\n";

    pthread_exit(NULL);
    return 0;
}