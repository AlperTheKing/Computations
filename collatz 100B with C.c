#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>  // For sysconf on POSIX systems
#endif

pthread_mutex_t mtx; // Mutex for protecting shared data

// ANSI color codes
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_YELLOW "\033[1;33m"
#define BOLD_BLUE "\033[1;34m"

// Function to calculate the number of Collatz steps for a given number
uint64_t collatz_steps(uint64_t n) {
    uint64_t steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

// Struct to pass data to threads
typedef struct {
    uint64_t start;
    uint64_t end;
    uint64_t max_steps;
    uint64_t number_with_max_steps;
} ThreadData;

// Function to find the number with maximum Collatz steps in a given range
void* find_max_collatz_steps_in_range(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    uint64_t local_max_steps = 0;
    uint64_t local_number_with_max_steps = data->start;

    for (uint64_t i = data->start; i < data->end; ++i) {
        uint64_t steps = collatz_steps(i);
        if (steps > local_max_steps) {
            local_max_steps = steps;
            local_number_with_max_steps = i;
        }
    }

    pthread_mutex_lock(&mtx);
    if (local_max_steps > data->max_steps) {
        data->max_steps = local_max_steps;
        data->number_with_max_steps = local_number_with_max_steps;
    }
    pthread_mutex_unlock(&mtx);

    return NULL;
}

// Get the number of CPU cores
int get_num_cores() {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

int main() {
    // Define the ranges, now including 10^11-10^12
    uint64_t ranges[12][2];
    uint64_t base = 1;  // Start from 10^0
    for (int i = 0; i <= 11; ++i) {
        ranges[i][0] = base;
        ranges[i][1] = base * 10;
        base *= 10;
    }

    pthread_mutex_init(&mtx, NULL);

    // Loop through each range and find the number with the maximum Collatz steps
    for (int r = 0; r < 12; ++r) {
        uint64_t start = ranges[r][0];
        uint64_t end = ranges[r][1];

        uint64_t max_steps = 0;
        uint64_t number_with_max_steps = start;

        // Start timing
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Split the range into chunks for parallel processing
        int num_threads = get_num_cores(); // Get number of available cores
        uint64_t chunk_size = (end - start) / num_threads;

        pthread_t threads[num_threads];
        ThreadData thread_data[num_threads];

        for (int i = 0; i < num_threads; ++i) {
            uint64_t chunk_start = start + i * chunk_size;
            uint64_t chunk_end = (i == num_threads - 1) ? end : chunk_start + chunk_size;

            thread_data[i].start = chunk_start;
            thread_data[i].end = chunk_end;
            thread_data[i].max_steps = max_steps;
            thread_data[i].number_with_max_steps = number_with_max_steps;

            pthread_create(&threads[i], NULL, find_max_collatz_steps_in_range, &thread_data[i]);
        }

        // Join the threads
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
            if (thread_data[i].max_steps > max_steps) {
                max_steps = thread_data[i].max_steps;
                number_with_max_steps = thread_data[i].number_with_max_steps;
            }
        }

        // End timing
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double execution_time = (end_time.tv_sec - start_time.tv_sec) +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

        // Display the result vertically with color
        printf(BOLD_BLUE "Range %lu - %lu:\n" RESET, start, end);
        printf(BOLD_GREEN "Number with max steps: %lu\n" RESET, number_with_max_steps);
        printf(BOLD_YELLOW "Steps: %lu\n" RESET, max_steps);
        printf(BOLD_RED "Time taken: %.6f seconds\n" RESET, execution_time);
        printf("-----------------------------\n");
    }

    pthread_mutex_destroy(&mtx);

    return 0;
}