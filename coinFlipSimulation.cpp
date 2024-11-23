#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include "pcg_random.hpp" // Include the PCG header file

// Total number of coin flips (10^11)
constexpr uint64_t NUM_FLIPS = 100000000000ULL;

// Number of predictions equal to NUM_FLIPS
constexpr uint64_t NUM_PREDICTIONS = NUM_FLIPS;

std::atomic<uint64_t> total_heads(0);
std::atomic<uint64_t> total_tails(0);
std::atomic<uint64_t> correct_predictions(0);

void flip_coins(uint64_t flips_per_thread, uint64_t seed) {
    pcg64 rng(seed);

    uint64_t local_heads = 0;
    uint64_t local_tails = 0;

    for (uint64_t i = 0; i < flips_per_thread; ++i) {
        uint64_t r = rng();
        if (r & 1) {
            ++local_heads;
        } else {
            ++local_tails;
        }
    }

    total_heads += local_heads;
    total_tails += local_tails;
}

void predict_outcomes(uint64_t predictions_per_thread, uint64_t seed, unsigned int thread_id) {
    pcg64 rng_prediction(seed);

    uint64_t local_correct_predictions = 0;
    uint64_t local_total_heads = 0;
    uint64_t local_total_tails = 0;

    // Initial guess
    uint64_t heads = total_heads.load();
    uint64_t tails = total_tails.load();
    char guess = (heads > tails) ? 'T' : 'H';

    for (uint64_t i = 0; i < predictions_per_thread; ++i) {
        // Recalculate the guess periodically to reduce atomic reads
        if (i % 1000000 == 0 && i != 0) {
            // Update the shared counts with local counts
            total_heads += local_total_heads;
            total_tails += local_total_tails;

            local_total_heads = 0;
            local_total_tails = 0;

            // Recalculate the guess
            heads = total_heads.load();
            tails = total_tails.load();
            guess = (heads > tails) ? 'T' : 'H';

            // Optional: Progress indicator
            // std::cout << "Thread " << thread_id << " processed " << i << " / " << predictions_per_thread << " predictions.\n";
        }

        // Generate next random bit
        uint64_t r = rng_prediction();
        char actual = (r & 1) ? 'H' : 'T';

        // Update local counts
        if (actual == 'H') {
            ++local_total_heads;
        } else {
            ++local_total_tails;
        }

        // Check if the guess was correct
        if (guess == actual) {
            ++local_correct_predictions;
        }
    }

    // Update shared counts and correct predictions after the loop
    total_heads += local_total_heads;
    total_tails += local_total_tails;
    correct_predictions += local_correct_predictions;
}

int main() {
    // Measure the total execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get the number of hardware threads
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    if (NUM_THREADS == 0) {
        NUM_THREADS = 1; // Fallback to 1 thread if hardware_concurrency can't detect
    }

    uint64_t flips_per_thread = NUM_FLIPS / NUM_THREADS;
    uint64_t remaining_flips = NUM_FLIPS % NUM_THREADS;

    // Generate high-quality seeds for each thread using std::random_device and std::seed_seq
    std::random_device rd;
    std::vector<uint32_t> seed_data(NUM_THREADS * 2); // Collect enough seed data
    for (auto& sd : seed_data) {
        sd = rd();
    }
    std::seed_seq seq(seed_data.begin(), seed_data.end());

    std::vector<uint64_t> thread_seeds(NUM_THREADS);
    seq.generate(thread_seeds.begin(), thread_seeds.end());

    std::vector<std::thread> threads;

    // Multithreaded coin flips
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        uint64_t seed = thread_seeds[i]; // Unique seed for each thread
        uint64_t flips = flips_per_thread + (i == 0 ? remaining_flips : 0);
        threads.emplace_back(flip_coins, flips, seed);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Initial Total Heads: " << total_heads.load() << "\n";
    std::cout << "Initial Total Tails: " << total_tails.load() << "\n";

    // Measure time after coin flips
    auto flip_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> flip_duration = flip_end_time - start_time;
    std::cout << "Time taken for coin flips: " << flip_duration.count() << " seconds\n";

    // Reset threads vector for predictions
    threads.clear();

    // Generate new seeds for prediction threads
    for (auto& sd : seed_data) {
        sd = rd();
    }
    std::seed_seq pred_seq(seed_data.begin(), seed_data.end());
    pred_seq.generate(thread_seeds.begin(), thread_seeds.end());

    // Predictions per thread
    uint64_t predictions_per_thread = NUM_PREDICTIONS / NUM_THREADS;
    uint64_t remaining_predictions = NUM_PREDICTIONS % NUM_THREADS;

    // Multithreaded predictions
    auto prediction_start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        uint64_t seed = thread_seeds[i]; // Unique seed for each thread
        uint64_t predictions = predictions_per_thread + (i == 0 ? remaining_predictions : 0);
        threads.emplace_back(predict_outcomes, predictions, seed, i + 1);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Measure time after predictions
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prediction_duration = end_time - prediction_start_time;
    std::chrono::duration<double> total_duration = end_time - start_time;

    double accuracy = (static_cast<double>(correct_predictions.load()) / NUM_PREDICTIONS) * 100.0;

    std::cout << "Final Total Heads: " << total_heads.load() << "\n";
    std::cout << "Final Total Tails: " << total_tails.load() << "\n";
    std::cout << "Correct Predictions: " << correct_predictions.load() << " out of " << NUM_PREDICTIONS << "\n";
    std::cout << "Prediction Accuracy: " << accuracy << "%\n";
    std::cout << "Time taken for predictions: " << prediction_duration.count() << " seconds\n";
    std::cout << "Total Execution Time: " << total_duration.count() << " seconds\n";

    return 0;
}