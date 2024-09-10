#include <iostream>
#include <pthread.h>
#include <random>
#include <vector>
#include <ctime>
#include <chrono>  // For wall-clock time measurement
#include <thread>  // For hardware_concurrency()
#include <random123/threefry.h>
#include <random123/philox.h>

#define BATCH_SIZE 100000000  // Process 100 million people at a time
#define NUM_BATCHES 100  // Number of batches

// Struct to hold each person's data
struct Person {
    int birth_year;
    int birth_month;
    int birth_day;
};

// Global variables for threading
int count_greater = 0;
long long total_count_greater = 0;  // Track total count across all batches
pthread_mutex_t count_mutex;
int NUM_THREADS;  // Number of threads dynamically set based on hardware

// Function to check if a year is a leap year
bool is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// Function to generate a person's random birth date
void generate_birth_date(Person &person, int thread_id, int person_index) {
    using rng_t = r123::Philox4x32; // Use random123's Philox
    rng_t rng;
    rng_t::ctr_type c = {{0}};
    rng_t::key_type k = {{static_cast<uint32_t>(thread_id)}};
    c[0] = person_index;
    rng_t::ctr_type r = rng(c, k);

    person.birth_year = r[0] % 101 + 1900;  // Years from 1900 to 2000 (age 0 to 100)
    person.birth_month = r[1] % 12 + 1;     // Months from 1 to 12

    // Determine the number of days in the birth month
    int days_in_month;
    switch (person.birth_month) {
        case 2:
            days_in_month = is_leap_year(person.birth_year) ? 29 : 28;  // February
            break;
        case 4: case 6: case 9: case 11:
            days_in_month = 30;  // April, June, September, November
            break;
        default:
            days_in_month = 31;  // All other months
    }

    // Generate a random day in the valid range for the month
    person.birth_day = r[2] % days_in_month + 1;
}

// Function for each thread to process its portion of the population
void *process_population(void *threadid) {
    long tid = (long) threadid;
    int local_count = 0;
    
    // Process a batch of people in each thread
    for (int i = tid; i < BATCH_SIZE; i += NUM_THREADS) {
        Person person;
        // Generate random birth date for each person
        generate_birth_date(person, tid, i);
        
        // Check if birth month is greater than the last digit of the birth year
        int last_digit_of_year = person.birth_year % 10;
        if (person.birth_month > last_digit_of_year) {
            local_count++;
        }
    }
    
    // Lock mutex before updating global count
    pthread_mutex_lock(&count_mutex);
    count_greater += local_count;
    pthread_mutex_unlock(&count_mutex);
    
    pthread_exit(nullptr);
}

int main() {
    // Dynamically set the number of threads based on system's hardware concurrency
    NUM_THREADS = std::thread::hardware_concurrency();
    if (NUM_THREADS == 0) {
        NUM_THREADS = 4;  // Fallback in case hardware_concurrency() fails
    }

    std::cout << "Using " << NUM_THREADS << " threads." << std::endl;
    
    // Initialize random123 engine and variables
    srand(time(0));
    
    // Thread-related variables
    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&count_mutex, nullptr);
    
    // Start time measurement using std::chrono
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process in batches
    for (int batch = 0; batch < NUM_BATCHES; ++batch) {
        count_greater = 0;  // Reset count for each batch

        // Create threads
        for (long t = 0; t < NUM_THREADS; ++t) {
            pthread_create(&threads[t], nullptr, process_population, (void *) t);
        }
    
        // Wait for all threads to finish
        for (long t = 0; t < NUM_THREADS; ++t) {
            pthread_join(threads[t], nullptr);
        }

        // Aggregate total valid counts across all batches
        pthread_mutex_lock(&count_mutex);
        total_count_greater += count_greater;
        pthread_mutex_unlock(&count_mutex);
    }
    
    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Calculate and print the aggregated ratio
    long long total_people = static_cast<long long>(NUM_BATCHES) * static_cast<long long>(BATCH_SIZE);  // Fix overflow
    double final_ratio = double(total_count_greater) / total_people;
    std::cout << "Final ratio of people where birth month is greater than the last digit of birth year: " << final_ratio << std::endl;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;
    
    // Clean up
    pthread_mutex_destroy(&count_mutex);
    
    return 0;
}