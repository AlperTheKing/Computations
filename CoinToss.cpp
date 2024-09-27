#include <iostream>
#include <pthread.h>
#include <chrono>
#include <random123/philox.h>  // Include Random123 library for philox
#include <thread>  // For std::thread::hardware_concurrency()
#include <cstdint>  // For uint32_t

using namespace std;
using namespace chrono;
using r123::Philox4x32;  // Using Philox RNG from Random123

// Constants
const unsigned long long SIMULATIONS = 100000000000ULL;  // 100 billion simulations
unsigned int numThreads = 8;  // Number of threads

// Thread-safe counters
unsigned long long HTTTH_count = 0;
unsigned long long HTHTH_count = 0;
pthread_mutex_t pthread_mutex = PTHREAD_MUTEX_INITIALIZER;

// Random123 typedefs
typedef Philox4x32 RNG;
RNG rng;

// Function to check for patterns (stopping condition)
bool checkPattern(uint32_t sequence, uint32_t pattern) {
    return (sequence & 0x1F) == pattern;  // Mask the last 5 bits and check for a match
}

// Function to simulate coin flips until one of the patterns appears
void* simulateFlips(void* arg) {
    unsigned long long localHTTTH_count = 0;
    unsigned long long localHTHTH_count = 0;

    // Set up Random123 generator
    uint32_t tid = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(arg));
    RNG::ctr_type counter = {{0, tid, 0, 0}};  // Each thread uses a different counter
    RNG::ukey_type key = {{0xdeadbeef}};  // Arbitrary key

    unsigned long long flipsPerThread = SIMULATIONS / numThreads;

    // Define patterns
    const uint32_t HTTTH_pattern = 0b01110;  // HTTTH in binary (H=0, T=1)
    const uint32_t HTHTH_pattern = 0b01010;  // HTHTH in binary (H=0, T=1)

    for (unsigned long long i = 0; i < flipsPerThread; ++i) {
        uint32_t sequence = 0;  // Store the last 5 flips

        // Keep flipping until one of the patterns appears
        while (true) {
            counter[0] = i;  // Increment the counter for each flip
            RNG::ctr_type result = rng(counter, key);

            // Simulate flipping and shifting the sequence (last 5 flips)
            sequence = ((sequence << 1) | (result[0] & 1)) & 0x1F;  // Shift left and add new flip

            // Check if the pattern "HTTTH" appears
            if (checkPattern(sequence, HTTTH_pattern)) {
                localHTTTH_count++;
                break;
            }
            // Check if the pattern "HTHTH" appears
            else if (checkPattern(sequence, HTHTH_pattern)) {
                localHTHTH_count++;
                break;
            }
        }
    }

    // Lock the mutex before updating the global count
    pthread_mutex_lock(&pthread_mutex);
    HTTTH_count += localHTTTH_count;
    HTHTH_count += localHTHTH_count;
    pthread_mutex_unlock(&pthread_mutex);

    return nullptr;
}

int main() {
    // Get the number of available threads
    numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 2;  // Fallback to 2 threads if hardware_concurrency() cannot determine
    }

    cout << "Using " << numThreads << " threads." << endl;

    // Time measurement start
    auto start = high_resolution_clock::now();

    // Create threads
    pthread_t threads[numThreads];
    for (unsigned int i = 0; i < numThreads; ++i) {
        pthread_create(&threads[i], nullptr, simulateFlips, (void*)(uintptr_t)i);
    }

    // Join threads
    for (unsigned int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Time measurement end
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    // Calculate results
    cout << "HTTTH count: " << HTTTH_count << endl;
    cout << "HTHTH count: " << HTHTH_count << endl;
    cout << "Total simulations: " << SIMULATIONS << endl;
    cout << "Time elapsed: " << elapsed.count() << " seconds." << endl;

    // Determine which pattern is more likely
    if (HTTTH_count > HTHTH_count) {
        cout << "HTTTH is more likely." << endl;
    } else if (HTHTH_count > HTTTH_count) {
        cout << "HTHTH is more likely." << endl;
    } else {
        cout << "Both are equally likely." << endl;
    }

    return 0;
}