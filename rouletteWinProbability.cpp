#include <iostream>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <random123/threefry.h>
#include <random123/philox.h>
#include <thread>  // For hardware concurrency

#define NUM_SPINS 1000000000  // 1 billion simulations
#define TOTAL_NUMBERS 37      // European Roulette: 0 to 36

// Red or Black lookup table for numbers 0-36 (-1: neither, 1: red, 0: black)
const int red_black[] = {
    -1, 1, 0, 1, 0, 1, 0, 1, 0, 1,   // 0-9
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,    // 10-19
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,    // 20-29
    1, 0, 1, 0, 1, 0, 1              // 30-36
};

// Struct for tracking the wins for each thread
struct ThreadResult {
    long long wins_straight;
    long long wins_red;
    long long wins_black;
    long long wins_odd;
    long long wins_even;
    long long wins_low;
    long long wins_high;
    long long wins_first_dozen;
    long long wins_second_dozen;
    long long wins_third_dozen;
    long long wins_first_column;
    long long wins_second_column;
    long long wins_third_column;
    long long total_spins;
};

// Function to simulate roulette bets for each thread
void* simulate_roulette(void* arg) {
    long thread_id = (long) arg;
    ThreadResult* result = new ThreadResult{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    using rng_t = r123::Philox4x32;  // Use random123 Philox generator
    rng_t rng;
    rng_t::ctr_type c = {{0}};
    rng_t::key_type k = {{static_cast<uint32_t>(thread_id)}};
    
    // Run the simulation
    for (long long i = 0; i < NUM_SPINS / std::thread::hardware_concurrency(); ++i) {
        c[0] = i;  // Counter for random123
        rng_t::ctr_type r = rng(c, k);  // Generate random numbers

        // Spin the wheel: get a random number between 0 and 36
        int spin_result = r[0] % TOTAL_NUMBERS;

        // Randomly select the number the player bets on
        int bet_number = r[1] % TOTAL_NUMBERS;

        // Check if the player wins by betting on a specific number
        if (bet_number == spin_result) {
            result->wins_straight++;
        }

        // Check for Red/Black
        if (spin_result > 0) {
            if (red_black[spin_result] == 1) {
                result->wins_red++;
            } else {
                result->wins_black++;
            }
        }

        // Check for Odd/Even
        if (spin_result > 0 && spin_result % 2 == 0) {
            result->wins_even++;
        } else if (spin_result > 0 && spin_result % 2 == 1) {
            result->wins_odd++;
        }

        // Check for High/Low (1-18 is low, 19-36 is high)
        if (spin_result >= 1 && spin_result <= 18) {
            result->wins_low++;
        } else if (spin_result >= 19 && spin_result <= 36) {
            result->wins_high++;
        }

        // Check for Dozen Bets
        if (spin_result >= 1 && spin_result <= 12) {
            result->wins_first_dozen++;
        } else if (spin_result >= 13 && spin_result <= 24) {
            result->wins_second_dozen++;
        } else if (spin_result >= 25 && spin_result <= 36) {
            result->wins_third_dozen++;
        }

        // Check for Column Bets (1st, 2nd, and 3rd column on the roulette board)
        if (spin_result > 0) {
            if ((spin_result - 1) % 3 == 0) {
                result->wins_first_column++;
            } else if ((spin_result - 2) % 3 == 0) {
                result->wins_second_column++;
            } else if ((spin_result - 3) % 3 == 0) {
                result->wins_third_column++;
            }
        }

        // Increment total spins
        result->total_spins++;
    }

    pthread_exit(result);
}

int main() {
    // Dynamically determine the number of threads
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;  // Fallback if hardware_concurrency fails
    }

    std::cout << "Using " << num_threads << " threads." << std::endl;

    // Initialize threads
    pthread_t threads[num_threads];

    // Start the clock for performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create and run threads
    for (long t = 0; t < num_threads; ++t) {
        pthread_create(&threads[t], nullptr, simulate_roulette, (void*) t);
    }

    // Collect results from each thread
    ThreadResult total_result{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (long t = 0; t < num_threads; ++t) {
        ThreadResult* thread_result;
        pthread_join(threads[t], (void**) &thread_result);

        // Aggregate results
        total_result.wins_straight += thread_result->wins_straight;
        total_result.wins_red += thread_result->wins_red;
        total_result.wins_black += thread_result->wins_black;
        total_result.wins_odd += thread_result->wins_odd;
        total_result.wins_even += thread_result->wins_even;
        total_result.wins_low += thread_result->wins_low;
        total_result.wins_high += thread_result->wins_high;
        total_result.wins_first_dozen += thread_result->wins_first_dozen;
        total_result.wins_second_dozen += thread_result->wins_second_dozen;
        total_result.wins_third_dozen += thread_result->wins_third_dozen;
        total_result.wins_first_column += thread_result->wins_first_column;
        total_result.wins_second_column += thread_result->wins_second_column;
        total_result.wins_third_column += thread_result->wins_third_column;
        total_result.total_spins += thread_result->total_spins;

        delete thread_result;
    }

    // End the clock
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Calculate and print probabilities as percentages
    std::cout << "Probability of winning by betting on a specific number (straight): "
              << static_cast<double>(total_result.wins_straight) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on red: "
              << static_cast<double>(total_result.wins_red) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on black: "
              << static_cast<double>(total_result.wins_black) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on odd: "
              << static_cast<double>(total_result.wins_odd) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on even: "
              << static_cast<double>(total_result.wins_even) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on low (1-18): "
              << static_cast<double>(total_result.wins_low) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on high (19-36): "
              << static_cast<double>(total_result.wins_high) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on first dozen (1-12): "
              << static_cast<double>(total_result.wins_first_dozen) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on second dozen (13-24): "
              << static_cast<double>(total_result.wins_second_dozen) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on third dozen (25-36): "
              << static_cast<double>(total_result.wins_third_dozen) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on first column: "
              << static_cast<double>(total_result.wins_first_column) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on second column: "
              << static_cast<double>(total_result.wins_second_column) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Probability of winning by betting on third column: "
              << static_cast<double>(total_result.wins_third_column) / total_result.total_spins * 100 << "%" << std::endl;

    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}