#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <atomic>

// Maximum values for a, b, c, d
#define MAX_A_B_C_D 10000

// Maximum value for e, calculated based on a, b, c, d <= 10,000
#define MAX_E 14142

// Minimum practical chunk size to prevent excessive overhead
#define MIN_CHUNK_SIZE 10

// Structure to hold a solution
struct Solution {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint64_t e;
};

// Structure to define a work chunk
struct WorkChunk {
    uint64_t e_start;
    uint64_t e_end;
};

// Global work chunks and mutex
std::vector<WorkChunk> workChunks;
std::mutex queueMutex;

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
               const std::vector<uint64_t>& i_pows, 
               const std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>& ab_pows_map, 
               std::vector<Solution>& cpuSolutions, 
               std::atomic<float>& cpuTime) 
{
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        WorkChunk chunk;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workChunks.empty()) {
                break;
            }
            chunk = workChunks.back();
            workChunks.pop_back();
        }

        for (uint64_t e = chunk.e_start; e <= chunk.e_end && e <= MAX_E; e++) {
            uint64_t e4 = i_pows[e]; // e^4

            // Iterate over c and d
            for (uint64_t c = 1; c <= MAX_A_B_C_D; c++) {
                uint64_t c4 = i_pows[c];
                if (c4 >= e4) break; // c^4 should be less than e^4

                for (uint64_t d = c; d <= MAX_A_B_C_D; d++) { // c <= d
                    uint64_t cd4 = c4 + i_pows[d];
                    if (cd4 >= e4) break; // c^4 + d^4 < e^4

                    uint64_t remaining = e4 - cd4;

                    // Find (a, b) pairs where a^4 + b^4 = remaining
                    auto it = ab_pows_map.find(remaining);
                    if (it != ab_pows_map.end()) {
                        for (const auto& pair : it->second) {
                            uint64_t a = pair.first;
                            uint64_t b = pair.second;

                            // Ensure a <= b <= c to satisfy a <= b <= c <= d
                            if (b <= c) {
                                Solution sol = {a, b, c, d, e};
                                cpuSolutions.push_back(sol);
                            }
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

    // Precompute i^4
    std::vector<uint64_t> i_pows(MAX_E + 1, 0);
    for (uint64_t i = 1; i <= MAX_E; i++) {
        i_pows[i] = i * i * i * i; // i^4
    }

    // Precompute all possible a^4 + b^4 and store in a hash map
    std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> ab_pows_map;
    ab_pows_map.reserve(static_cast<size_t>(MAX_A_B_C_D) * static_cast<size_t>(MAX_A_B_C_D) / 10); // Reserve space to reduce rehashing

    for (uint64_t a = 1; a <= MAX_A_B_C_D; a++) {
        uint64_t a4 = i_pows[a];
        for (uint64_t b = a; b <= MAX_A_B_C_D; b++) { // Ensure a <= b to reduce duplicates
            uint64_t sum = a4 + i_pows[b];
            if (sum > i_pows[MAX_E]) break; // Do not store sums greater than e^4
            ab_pows_map[sum].emplace_back(a, b);
        }
    }

    std::cout << "Precomputed all possible a^4 + b^4 sums." << std::endl;

    // Determine the number of CPU threads to use (e.g., number of hardware threads)
    unsigned int numCPUs = std::thread::hardware_concurrency();
    if (numCPUs == 0) numCPUs = 4; // Fallback to 4 if unable to detect
    std::cout << "Number of CPU threads to use: " << numCPUs << std::endl;

    // Initialize work queue with dynamically calculated chunk size
    initializeWorkQueue(MAX_E, numCPUs);

    // Vectors to store solutions and computation times per CPU thread
    std::vector<std::vector<Solution>> allCpuSolutions(numCPUs);
    std::vector<std::atomic<float>> cpuTimesAtomic(numCPUs);
    for (auto& time : cpuTimesAtomic) {
        time = 0.0f;
    }

    // Create worker threads for CPU
    std::vector<std::thread> cpuWorkers;
    for (unsigned int i = 0; i < numCPUs; i++) {
        cpuWorkers.emplace_back(cpuWorker, i, std::cref(i_pows), std::cref(ab_pows_map), std::ref(allCpuSolutions[i]), std::ref(cpuTimesAtomic[i]));
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
    combinedSolutions.reserve(100); // Initial reserve; adjust based on expected number of solutions

    for (unsigned int i = 0; i < numCPUs; i++) {
        combinedSolutions.insert(combinedSolutions.end(), allCpuSolutions[i].begin(), allCpuSolutions[i].end());
    }

    // Sort the combined solutions in ascending order based on e
    std::sort(combinedSolutions.begin(), combinedSolutions.end(), [](const Solution& a, const Solution& b) -> bool {
        return a.e < b.e;
    });

    // Display results
    uint64_t totalSolutions = combinedSolutions.size();
    std::cout << "Total solutions found: " << totalSolutions << std::endl;

    uint64_t solutionNumber = 1;
    for (const auto& sol : combinedSolutions) {
        std::cout << "Solution " << solutionNumber << ": " 
                  << sol.a << "^4 + " 
                  << sol.b << "^4 + " 
                  << sol.c << "^4 + " 
                  << sol.d << "^4 = " 
                  << sol.e << "^4" << std::endl;
        solutionNumber++;
    }

    // Display computation times in seconds
    for (unsigned int i = 0; i < numCPUs; i++) {
        std::cout << "CPU Thread " << i << " found " << allCpuSolutions[i].size() 
                  << " solutions in " << cpuTimesAtomic[i] << " seconds." << std::endl;
    }

    std::cout << "Total computation time: " << total_duration.count() << " seconds." << std::endl;
    std::cout << "Computation completed successfully." << std::endl;
    return 0;
}