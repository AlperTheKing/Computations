#include <iostream>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <thread>  // For hardware_concurrency()
#include "random123/xoshiro512starstar.h"  // Include the xoshiro512starstar.h header

#define NUM_SIMULATIONS 100000000000  // 100 billion simulations

using namespace std;
using namespace std::chrono;

// Splitmix64 function to seed the xoshiro512**
uint64_t splitmix64(uint64_t *seed) {
    uint64_t result = (*seed += 0x9E3779B97F4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

// Seed the xoshiro512** generator using splitmix64
void seed_xoshiro512(uint64_t seed) {
    for (int i = 0; i < 8; i++) {
        s[i] = splitmix64(&seed);
    }
}

// Thread data structure
struct ThreadData {
    int id;
    int num_simulations;
    uint64_t red_red_count;  // Red first, then red again
    uint64_t red_green_count;  // Red first, then green
    uint64_t all_red_urn_selected;  // Track how often the urn with 99 red balls is selected
};

// Function to simulate the urn problem and track red-green outcomes
void* simulate_urns(void* threadarg) {
    ThreadData* data = (ThreadData*)threadarg;

    uint64_t local_red_red_count = 0;
    uint64_t local_red_green_count = 0;
    uint64_t local_all_red_urn_selected = 0;

    for (uint64_t i = 0; i < data->num_simulations; i++) {
        // Pick a random urn using next() from xoshiro512starstar.h
        uint64_t urn_choice = next() % 100;  // Call the next random number from xoshiro512**

        if (urn_choice == 99) {
            // Track the selection of the urn with all red balls
            local_all_red_urn_selected++;

            // If urn 99 is picked (all red urn), the next ball will definitely be red
            local_red_red_count++;
        } else {
            // If any of the other 99 urns is picked, 1 red ball, 98 green
            uint64_t ball_choice = next() % 99;  // Use next() again for ball choice
            if (ball_choice == 0) {
                // Red ball chosen on the first draw, next ball must be green
                local_red_green_count++;
            }
        }
    }

    data->red_red_count = local_red_red_count;
    data->red_green_count = local_red_green_count;
    data->all_red_urn_selected = local_all_red_urn_selected;
    pthread_exit(NULL);
}

int main() {
    // Initialize the state (s) using the seeding function provided
    seed_xoshiro512(123456789);  // You can use any 64-bit seed value here

    // Get the number of hardware threads available on the system
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;  // Default to 4 threads if hardware_concurrency() fails
    }

    cout << "Using " << num_threads << " threads.\n";

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];  // Define array of ThreadData
    uint64_t total_red_red_count = 0;
    uint64_t total_red_green_count = 0;
    uint64_t total_all_red_urn_selected = 0;

    // Split the total simulations equally across all threads
    int simulations_per_thread = NUM_SIMULATIONS / num_threads;

    auto start = high_resolution_clock::now();

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].id = i;
        thread_data[i].num_simulations = simulations_per_thread;
        thread_data[i].red_red_count = 0;
        thread_data[i].red_green_count = 0;
        thread_data[i].all_red_urn_selected = 0;
        int rc = pthread_create(&threads[i], NULL, simulate_urns, (void*)&thread_data[i]);
        if (rc) {
            cout << "Error: Unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    // Join threads and accumulate results
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_red_red_count += thread_data[i].red_red_count;
        total_red_green_count += thread_data[i].red_green_count;
        total_all_red_urn_selected += thread_data[i].all_red_urn_selected;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Output results
    uint64_t total_red_count = total_red_red_count + total_red_green_count;
    double prob_red = (double)total_red_red_count / total_red_count;
    double prob_green = (double)total_red_green_count / total_red_count;

    cout << "Total red balls drawn: " << total_red_count << "\n";
    cout << "Red followed by Red: " << total_red_red_count << "\n";
    cout << "Red followed by Green: " << total_red_green_count << "\n";
    cout << "Urn with all red balls selected: " << total_all_red_urn_selected << " times\n";
    cout << "Probability of next ball being Red: " << prob_red << "\n";
    cout << "Probability of next ball being Green: " << prob_green << "\n";
    cout << "Time taken: " << duration.count() << " seconds\n";

    pthread_exit(NULL);
    return 0;
}