#include <iostream>
#include <pthread.h>
#include <cmath>
#include <chrono>
#include <vector>

using namespace std;

const int NUM_THREADS = 8;
pthread_mutex_t mutexT;
int tatamiCount = 0;
bool found = false;
int resultSize = -1;

// Function to determine if a room size is tatami-free based on refined rules
bool isTatamiFree(int a, int b) {
    // Both dimensions must be even
    if (a % 2 == 0 && b % 2 == 0) {
        // Now we need to ensure it's not possible to tile the room with the mats
        // based on the "no four corners meet" rule.
        // Further pattern checks should be added based on specific layout restrictions.
        // Example logic for pattern detection:
        if ((a % 4 == 0 && b % 4 != 0) || (b % 4 == 0 && a % 4 != 0)) {
            return true;
        }
    }
    return false;
}

// Thread function to calculate number of tatami-free divisor pairs
void *calculateTatamiFreeRooms(void *thread_id) {
    long tid = (long) thread_id;

    for (int s = tid; s <= 100000000; s += NUM_THREADS) {
        int count = 0;
        
        // Find all divisors of s and check if they form tatami-free rooms
        for (int a = 2; a <= sqrt(s); ++a) {
            if (s % a == 0) {
                int b = s / a;
                if (isTatamiFree(a, b)) {
                    count++;
                }
            }
        }

        // Update shared variables in a thread-safe manner
        pthread_mutex_lock(&mutexT);
        if (count >= 200 && !found) {
            found = true;
            resultSize = s;
        }
        pthread_mutex_unlock(&mutexT);

        if (found) break;
    }
    pthread_exit(NULL);
}

int main() {
    // Start time
    auto start = chrono::high_resolution_clock::now();

    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&mutexT, NULL);

    // Create threads to calculate tatami-free rooms
    for (long t = 0; t < NUM_THREADS; ++t) {
        pthread_create(&threads[t], NULL, calculateTatamiFreeRooms, (void *)t);
    }

    // Join threads
    for (long t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&mutexT);

    // End time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Output result
    if (resultSize != -1) {
        cout << "Found at size: " << resultSize << endl;
    } else {
        cout << "No result found." << endl;
    }

    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}