#include "random123/xoshiro256plusplus.h"  // xoshiro256++ header file
#include <iostream>
#include <random>
#include <chrono>
#include <set>
#include <pthread.h>
#include <cstdint>
#include <cmath>
#include <thread>  // std::thread için eklenmeli

// Maximum simulated number size (approximation for large numbers)
const uint64_t MAX_VALUE = static_cast<uint64_t>(pow(10, 6));

// Number of simulations to run in total
const int SIMULATIONS = 1000000000;

// Global result counters
long long alice_wins = 0;
long long bob_wins = 0;

// Mutex for thread-safe updates of global results
pthread_mutex_t result_mutex;

// Function to generate a simulated prime factor set for a large random number
std::set<uint64_t> generate_random_prime_factors() {
    std::set<uint64_t> prime_factors;
    std::uniform_int_distribution<uint64_t> prime_dist(2, MAX_VALUE);

    // Simulate number of prime factors (assume average of 5 factors)
    std::random_device rd;
    std::mt19937 generator(rd());
    std::poisson_distribution<int> factor_count(5);
    int num_factors = factor_count(generator);

    for (int i = 0; i < num_factors; ++i) {
        prime_factors.insert(prime_dist(generator));
    }

    return prime_factors;
}

// Function to check if two numbers have a common prime factor
bool have_common_prime_factors(const std::set<uint64_t>& factors_a, const std::set<uint64_t>& factors_b) {
    for (const auto& factor : factors_a) {
        if (factors_b.find(factor) != factors_b.end()) {
            return true;  // Common factor found
        }
    }
    return false;
}

// Data structure to pass data to each thread
struct ThreadData {
    int simulations_per_thread;
    uint64_t seed;
};

// Thread function to perform the simulation
void* run_simulation(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int num_simulations = data->simulations_per_thread;
    uint64_t seed = data->seed;

    // Initialize xoshiro256++ random number generator
    s[0] = seed;
    s[1] = seed + 1;
    s[2] = seed + 2;
    s[3] = seed + 3;

    long long local_alice_wins = 0;
    long long local_bob_wins = 0;

    for (int i = 0; i < num_simulations; ++i) {
        std::set<uint64_t> factors_a = generate_random_prime_factors();
        std::set<uint64_t> factors_b = generate_random_prime_factors();

        if (have_common_prime_factors(factors_a, factors_b)) {
            local_alice_wins++;
        } else {
            local_bob_wins++;
        }
    }

    // Update global results with mutex protection
    pthread_mutex_lock(&result_mutex);
    alice_wins += local_alice_wins;
    bob_wins += local_bob_wins;
    pthread_mutex_unlock(&result_mutex);

    return nullptr;
}

int main() {
    // Initialize the random number generator
    std::random_device rd;

    // Include the thread header
    #include <thread>

    // Determine the number of threads to use
    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads." << std::endl;

    // Divide the number of simulations across the threads
    int simulations_per_thread = SIMULATIONS / num_threads;

    // Initialize pthreads and thread data
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Start timing the execution
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize mutex for result protection
    pthread_mutex_init(&result_mutex, nullptr);

    // Create and run threads
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].simulations_per_thread = simulations_per_thread;
        thread_data[i].seed = rd();  // Each thread gets a unique seed
        pthread_create(&threads[i], nullptr, run_simulation, (void*)&thread_data[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Destroy the mutex
    pthread_mutex_destroy(&result_mutex);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the results
    std::cout << "Total Alice wins: " << alice_wins << std::endl;
    std::cout << "Total Bob wins: " << bob_wins << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}