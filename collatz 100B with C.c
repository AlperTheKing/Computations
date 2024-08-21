#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#define CACHE_SIZE 1000000
#define THREAD_COUNT 16

typedef struct {
    long long start;
    long long end;
    long long number_with_max_steps;
    int max_steps;
} RangeData;

int* memoization_cache;

// Optimized function to calculate the number of steps for the Collatz sequence with memoization
int collatz_steps(long long n) {
    int steps = 0;
    long long original_n = n;

    while (n != 1) {
        if (n < CACHE_SIZE && memoization_cache[n] != -1) {
            steps += memoization_cache[n];
            break;
        }

        if (n % 2 == 0) {
            n = n >> 1; // Use bitwise operation instead of division
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }

    if (original_n < CACHE_SIZE) {
        memoization_cache[original_n] = steps;
    }
    
    return steps;
}

// Function to calculate the max steps in a given range
void* calculate_collatz_range(void* arg) {
    RangeData* data = (RangeData*)arg;
    data->max_steps = 0;
    data->number_with_max_steps = 0;

    for (long long i = data->start; i < data->end; ++i) {
        int steps = collatz_steps(i);
        if (steps > data->max_steps) {
            data->max_steps = steps;
            data->number_with_max_steps = i;
        }
    }

    return NULL;
}

// Function to split the range into smaller subranges for threads
void split_range(long long start, long long end, int num_splits, RangeData ranges[]) {
    long long step = (end - start) / num_splits;
    
    for (int i = 0; i < num_splits; ++i) {
        ranges[i].start = start + i * step;
        ranges[i].end = (i == num_splits - 1) ? end : ranges[i].start + step;
    }
}

int main() {
    // Allocate and initialize the memoization cache
    memoization_cache = (int*)malloc(CACHE_SIZE * sizeof(int));
    for (int i = 0; i < CACHE_SIZE; ++i) {
        memoization_cache[i] = -1;
    }

    RangeData groups[] = {
        {1, 10, 0, 0},
        {10, 100, 0, 0},
        {100, 1000, 0, 0},
        {1000, 10000, 0, 0},
        {10000, 100000, 0, 0},
        {100000, 1000000, 0, 0},
        {1000000, 10000000, 0, 0},
        {10000000, 100000000, 0, 0},
        {100000000, 1000000000, 0, 0},
        {1000000000, 10000000000, 0, 0},
        {10000000000, 100000000000, 0, 0}
    };

    pthread_t threads[THREAD_COUNT];
    RangeData thread_data[THREAD_COUNT];

    for (int i = 0; i < sizeof(groups) / sizeof(groups[0]); ++i) {
        long long start = groups[i].start;
        long long end = groups[i].end;

        // Start timing the calculation
        clock_t begin = clock();

        // Split the range into subranges for each thread
        split_range(start, end, THREAD_COUNT, thread_data);

        // Start the threads
        for (int j = 0; j < THREAD_COUNT; ++j) {
            pthread_create(&threads[j], NULL, calculate_collatz_range, &thread_data[j]);
        }

        // Wait for all threads to finish
        for (int j = 0; j < THREAD_COUNT; ++j) {
            pthread_join(threads[j], NULL);
        }

        // Collect and find the maximum result
        long long number_with_max_steps = 0;
        int max_steps = 0;
        for (int j = 0; j < THREAD_COUNT; ++j) {
            if (thread_data[j].max_steps > max_steps) {
                max_steps = thread_data[j].max_steps;
                number_with_max_steps = thread_data[j].number_with_max_steps;
            }
        }

        // End timing the calculation
        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - begin) / CLOCKS_PER_SEC;

        // Print the results
        printf("Range %lld to %lld:\n", start, end);
        printf("  Number with max steps: %lld (%d steps)\n", number_with_max_steps, max_steps);
        printf("Calculation time: %.2f seconds\n\n", elapsed_time);
    }

    // Free the memoization cache
    free(memoization_cache);

    return 0;
}