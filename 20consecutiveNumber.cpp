#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <cmath>

// Define the set of primes
const std::vector<long long> primes = {2, 3, 5, 7, 11, 13};

// Configuration Constants
const long long MAX_N = 1000000000000000LL; // 10^15
const long long WINDOW_SIZE = 21;
const long long CHUNK_SIZE = 10000000000LL; // 1e10, adjustable based on system memory
const int NUM_THREADS = std::thread::hardware_concurrency();

// Mutexes for synchronized output and file writing
std::mutex cout_mutex;
std::mutex file_mutex;

// Output file to store all sequences
std::ofstream output_file("all_sequences_21.txt");

// Structure to hold a sequence result
struct Sequence {
    long long start;
    std::vector<long long> numbers;
};

// Worker function for each thread
void find_all_consecutive(long long start_num, long long end_num, std::vector<Sequence> &local_sequences) {
    long long current_count = 0;
    long long sequence_start = 0;

    for (long long n = start_num; n <= end_num; ++n) {
        bool divisible = false;
        for (const auto &p : primes) {
            if (n % p == 0) {
                divisible = true;
                break;
            }
        }

        if (divisible) {
            if (current_count == 0) {
                sequence_start = n;
            }
            current_count++;
            if (current_count == WINDOW_SIZE) {
                // Record the sequence
                Sequence seq;
                seq.start = sequence_start;
                for (long long i = 0; i < WINDOW_SIZE; ++i) {
                    seq.numbers.push_back(sequence_start + i);
                }
                local_sequences.push_back(seq);
                // Reset the count to continue searching for overlapping sequences
                current_count--;
                sequence_start += 1;
            }
        } else {
            current_count = 0;
        }
    }
}

int main() {
    std::cout << "Starting search for all sequences of " << WINDOW_SIZE << " consecutive numbers divisible by at least one prime in {2,3,5,7,11,13} up to " << MAX_N << ".\n";
    std::cout << "Using " << NUM_THREADS << " threads.\n";

    // Initialize timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize the output file
    output_file << "All sequences of " << WINDOW_SIZE << " consecutive numbers divisible by at least one prime in {2,3,5,7,11,13} up to " << MAX_N << ":\n\n";

    // Divide the range into chunks
    std::vector<std::pair<long long, long long>> chunks;
    for (long long start = 2; start <= MAX_N; start += CHUNK_SIZE) {
        long long end = start + CHUNK_SIZE - 1;
        if (end > MAX_N) end = MAX_N;
        chunks.emplace_back(std::make_pair(start, end));
    }

    // Atomic counter for progress reporting
    std::atomic<long long> chunks_processed(0);
    long long total_chunks = chunks.size();

    // Lambda function for thread workers
    auto worker = [&](int thread_id) {
        // Each thread will collect its own sequences to minimize locking
        std::vector<Sequence> local_sequences;

        while (true) {
            // Fetch the next chunk to process
            std::pair<long long, long long> chunk;
            {
                static std::mutex chunk_mutex;
                std::lock_guard<std::mutex> lock(chunk_mutex);
                if (chunks.empty()) break;
                chunk = chunks.back();
                chunks.pop_back();
            }

            long long start_num = chunk.first;
            long long end_num = chunk.second;

            // To handle overlapping sequences across chunks, consider the last (WINDOW_SIZE -1) numbers from the previous chunk
            if (start_num != 2) { // Not the first chunk
                start_num -= (WINDOW_SIZE - 1);
                if (start_num < 2) start_num = 2;
            }

            find_all_consecutive(start_num, end_num, local_sequences);
            chunks_processed++;

            // Progress reporting
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Thread " << thread_id << " processed chunk " << chunks_processed.load() << " / " << total_chunks << ".\n";
            }
        }

        // Write local sequences to the output file
        if (!local_sequences.empty()) {
            std::lock_guard<std::mutex> lock(file_mutex);
            for (const auto &seq : local_sequences) {
                output_file << "Sequence starting at: " << seq.start << "\nSequence:\n";
                for (const auto &num : seq.numbers) {
                    output_file << num << " ";
                }
                output_file << "\n\n";
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker, i + 1);
    }

    // Join threads
    for (auto &t : threads) {
        t.join();
    }

    // Close the output file
    output_file.close();

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Final output
    std::cout << "Search completed.\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds.\n";
    std::cout << "All sequences have been written to 'all_sequences_21.txt'.\n";

    return 0;
}