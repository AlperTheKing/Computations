#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <string>
#include <fstream>

// Maximum values for a, b, c, d
#define MAX_A_B_C_D 90000

// Maximum value for e, calculated based on a, b, c, d <= 90,000
#define MAX_E 127279

// Minimum practical chunk size to prevent excessive overhead
#define MIN_CHUNK_SIZE 10

// Structure to hold a solution
struct Solution {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    __int128 e;
};

// Structure to define a work chunk
struct WorkChunk {
    uint64_t e_start;
    uint64_t e_end;
};

// Structure to hold a^4 + b^4 sums
struct ABSum {
    uint64_t sum; // Using uint64_t since sum <= 1.3122e19 < 1.8446744e19
    uint64_t a;
    uint64_t b;
};

// Comparator struct to handle both (ABSum, __int128) and (__int128, ABSum)
struct CompareABSumWithInt128 {
    using is_transparent = void; // Enables heterogeneous lookup

    bool operator()(const ABSum& absum, const __int128& value) const {
        return static_cast<__int128>(absum.sum) < value;
    }
    bool operator()(const __int128& value, const ABSum& absum) const {
        return value < static_cast<__int128>(absum.sum);
    }
};

// Global work chunks replaced by an atomic index
std::atomic<size_t> currentChunkIndex(0);
std::vector<WorkChunk> workChunks;

// Function to convert __int128 to string for printing
std::string int128ToString(__int128 n) {
    bool isNegative = false;
    if (n < 0) {
        isNegative = true;
        n = -n;
    }
    std::string s;
    if (n == 0) {
        s = "0";
    } else {
        while (n > 0) {
            char digit = '0' + (n % 10);
            s.insert(s.begin(), digit);
            n /= 10;
        }
    }
    if (isNegative) {
        s.insert(s.begin(), '-');
    }
    return s;
}

// Function to initialize work queue with dynamically calculated chunk size
void initializeWorkQueue(uint64_t max_e, int num_threads) {
    uint64_t chunk_size = std::max(max_e / static_cast<uint64_t>(num_threads * 10), static_cast<uint64_t>(MIN_CHUNK_SIZE));
    uint64_t current_e = 1;
    while (current_e <= max_e) {
        WorkChunk chunk;
        chunk.e_start = current_e;
        chunk.e_end = std::min(current_e + chunk_size - 1, max_e);
        workChunks.push_back(chunk);
        current_e += chunk_size;
    }
    std::cout << "Total work chunks: " << workChunks.size() << " with chunk size: " << chunk_size << std::endl;
}

// Worker function for each CPU thread
void cpuWorker(int thread_id, 
               const std::vector<__int128>& i_pows, 
               const std::vector<ABSum>& ab_pows_sorted, 
               std::vector<Solution>& cpuSolutions, 
               std::atomic<float>& cpuTime) 
{
    auto start = std::chrono::high_resolution_clock::now();
    CompareABSumWithInt128 cmp;

    while (true) {
        size_t chunkIdx = currentChunkIndex.fetch_add(1);
        if (chunkIdx >= workChunks.size()) {
            break;
        }
        WorkChunk chunk = workChunks[chunkIdx];

        for (uint64_t e = chunk.e_start; e <= chunk.e_end && e <= MAX_E; e++) {
            __int128 e4 = i_pows[e]; // e^4

            // Iterate over c and d
            for (uint64_t c = 1; c <= MAX_A_B_C_D; c++) {
                __int128 c4 = i_pows[c];
                if (c4 >= e4) break; // c^4 should be less than e^4

                for (uint64_t d = c; d <= MAX_A_B_C_D; d++) { // c <= d
                    __int128 cd4 = c4 + i_pows[d];
                    if (cd4 >= e4) break; // c^4 + d^4 < e^4

                    __int128 remaining = e4 - cd4;

                    // Use equal_range with the transparent comparator
                    auto range = std::equal_range(ab_pows_sorted.begin(), ab_pows_sorted.end(), remaining, cmp);

                    for (auto it = range.first; it != range.second; ++it) {
                        uint64_t a = it->a;
                        uint64_t b = it->b;

                        // Ensure a <= b <= c to satisfy a <= b <= c <= d
                        if (b <= c) {
                            Solution sol = {a, b, c, d, static_cast<__int128>(e)};
                            cpuSolutions.push_back(sol);
                        }
                    }
                }
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = stop - start;
    cpuTime += duration.count();
}

int main() {
    // Start total computation time
    auto total_start = std::chrono::high_resolution_clock::now();

    // Precompute i^4 using __int128
    std::vector<__int128> i_pows(MAX_E + 1, 0);
    for (uint64_t i = 1; i <= MAX_E; i++) {
        i_pows[i] = static_cast<__int128>(i) * i * i * i; // i^4
    }

    std::cout << "Precomputed all i^4 values up to e = " << MAX_E << "." << std::endl;

    // Precompute all possible a^4 + b^4 and store in a sorted vector
    std::cout << "Starting precomputation of a^4 + b^4 sums..." << std::endl;

    // Multithreaded precomputation
    unsigned int numCPUs = std::thread::hardware_concurrency();
    if (numCPUs == 0) numCPUs = 4; // Fallback to 4 if unable to detect
    std::cout << "Number of CPU threads for precomputation: " << numCPUs << std::endl;

    // Split the range of a among threads
    std::vector<std::vector<ABSum>> thread_ab_pows(numCPUs);
    std::vector<std::thread> precomputeThreads;
    uint64_t a_per_thread = MAX_A_B_C_D / numCPUs;

    for (unsigned int i = 0; i < numCPUs; i++) {
        uint64_t a_start = i * a_per_thread + 1;
        uint64_t a_end = (i == numCPUs - 1) ? MAX_A_B_C_D : (i + 1) * a_per_thread;
        precomputeThreads.emplace_back([a_start, a_end, &i_pows, &thread_ab_pows, i]() {
            for (uint64_t a = a_start; a <= a_end; a++) {
                uint64_t a4 = static_cast<uint64_t>(i_pows[a]); // Cast to uint64_t
                for (uint64_t b = a; b <= MAX_A_B_C_D; b++) { // Ensure a <= b to reduce duplicates
                    uint64_t sum = static_cast<uint64_t>(i_pows[a]) + static_cast<uint64_t>(i_pows[b]);
                    if (sum > static_cast<uint64_t>(i_pows[MAX_E])) break; // Do not store sums greater than e^4
                    thread_ab_pows[i].push_back(ABSum{sum, a, b});
                }
                if (a % 10000 == 0) {
                    std::cout << "Precomputed a^4 + b^4 for a = " << a << " by thread " << i << std::endl;
                }
            }
        });
    }

    // Wait for all precompute threads to finish
    for (auto& th : precomputeThreads) {
        th.join();
    }

    std::cout << "Precomputed all possible a^4 + b^4 sums." << std::endl;

    // Merge all thread_ab_pows into a single vector
    std::cout << "Merging precomputed a^4 + b^4 sums from all threads..." << std::endl;
    std::vector<ABSum> ab_pows_vec;
    // Estimated reserve size: Number of pairs * sizeof(ABSum)
    // Note: Reserving this much memory may not be feasible; monitor memory usage
    // ab_pows_vec.reserve(static_cast<size_t>(MAX_A_B_C_D) * 100); // Previously set to 4.05e9

    // Instead of reserving, append all thread_ab_pows to ab_pows_vec
    for (unsigned int i = 0; i < numCPUs; i++) {
        ab_pows_vec.insert(ab_pows_vec.end(), thread_ab_pows[i].begin(), thread_ab_pows[i].end());
        // Clear the thread's local vector to free memory
        thread_ab_pows[i].clear();
        std::cout << "Thread " << i << " contributed " << ab_pows_vec.size() << " sums so far." << std::endl;
    }

    std::cout << "Total a^4 + b^4 sums: " << ab_pows_vec.size() << std::endl;

    // Sort the ab_pows_vec based on sum to enable binary search
    std::cout << "Sorting the a^4 + b^4 sums for binary search..." << std::endl;
    std::sort(ab_pows_vec.begin(), ab_pows_vec.end(), [](const ABSum& lhs, const ABSum& rhs) -> bool {
        return lhs.sum < rhs.sum;
    });

    std::cout << "Sorted the a^4 + b^4 sums for binary search." << std::endl;

    // Determine the number of CPU threads to use for the main computation
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback to 4 if unable to detect
    std::cout << "Number of CPU threads to use for main computation: " << numThreads << std::endl;

    // Initialize work queue with dynamically calculated chunk size
    initializeWorkQueue(MAX_E, numThreads);

    // Vectors to store solutions and computation times per CPU thread
    std::vector<std::vector<Solution>> allCpuSolutions(numThreads);
    std::vector<std::atomic<float>> cpuTimesAtomic(numThreads);
    for (auto& time : cpuTimesAtomic) {
        time = 0.0f;
    }

    // Create worker threads for CPU
    std::vector<std::thread> cpuWorkers;
    for (unsigned int i = 0; i < numThreads; i++) {
        cpuWorkers.emplace_back([i, &i_pows, &ab_pows_vec, &allCpuSolutions, &cpuTimesAtomic]() {
            cpuWorker(i, i_pows, ab_pows_vec, allCpuSolutions[i], cpuTimesAtomic[i]);
        });
    }

    // Wait for all CPU workers to finish
    for (auto& worker : cpuWorkers) {
        worker.join();
    }

    // Record total computation end time
    auto total_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> total_duration = total_stop - total_start;

    // Aggregate all solutions into a single vector for sorting
    std::vector<Solution> combinedSolutions;
    // Estimating an upper bound; adjust based on expected number of solutions
    combinedSolutions.reserve(1000); 

    for (unsigned int i = 0; i < numThreads; i++) {
        combinedSolutions.insert(combinedSolutions.end(), allCpuSolutions[i].begin(), allCpuSolutions[i].end());
    }

    std::cout << "Aggregated all solutions from threads." << std::endl;

    // Sort the combined solutions in ascending order based on e
    std::sort(combinedSolutions.begin(), combinedSolutions.end(), [](const Solution& a, const Solution& b) -> bool {
        return a.e < b.e;
    });

    // Display results
    uint64_t totalSolutions = combinedSolutions.size();
    std::cout << "Total solutions found: " << totalSolutions << std::endl;

    // Optionally, write solutions to a file to handle large output
    std::ofstream outfile("solutions.txt");
    if (outfile.is_open()) {
        uint64_t solutionNumber = 1;
        for (const auto& sol : combinedSolutions) {
            outfile << "Solution " << solutionNumber << ": " 
                    << sol.a << "^4 + " 
                    << sol.b << "^4 + " 
                    << sol.c << "^4 + " 
                    << sol.d << "^4 = " 
                    << int128ToString(sol.e) << "^4\n";
            solutionNumber++;
        }
        outfile.close();
        std::cout << "All solutions have been written to solutions.txt" << std::endl;
    } else {
        // If unable to open file, print to console (not recommended for large outputs)
        uint64_t solutionNumber = 1;
        for (const auto& sol : combinedSolutions) {
            std::cout << "Solution " << solutionNumber << ": " 
                      << sol.a << "^4 + " 
                      << sol.b << "^4 + " 
                      << sol.c << "^4 + " 
                      << sol.d << "^4 = " 
                      << int128ToString(sol.e) << "^4" << std::endl;
            solutionNumber++;
        }
    }

    // Display computation times in seconds
    for (unsigned int i = 0; i < numThreads; i++) {
        std::cout << "CPU Thread " << i << " found " << allCpuSolutions[i].size() 
                  << " solutions in " << cpuTimesAtomic[i] << " seconds." << std::endl;
    }

    std::cout << "Total computation time: " << total_duration.count() << " seconds." << std::endl;
    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}