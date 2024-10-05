#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <cmath>

using namespace std;

// Precompute factorials of digits 0 to 9
vector<int> factorials = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};

// Mutex for synchronized access to shared cache
mutex cache_mutex;

// Function to compute the sum of the factorial of digits of a number
int sumOfFactorials(int n) {
    int sum = 0;
    while (n > 0) {
        sum += factorials[n % 10];
        n /= 10;
    }
    return sum;
}

// Function to compute chain starting from a number
int computeChain(int n, unordered_map<int, int>& cache) {
    vector<int> chain;
    unordered_set<int> seen;

    int current = n;
    while (true) {
        {
            // Check if current number's chain length is already computed
            lock_guard<mutex> lock(cache_mutex);
            if (cache.find(current) != cache.end()) {
                int totalLength = chain.size() + cache[current];
                // Update cache for all numbers in the current chain
                for (size_t i = 0; i < chain.size(); ++i) {
                    cache[chain[i]] = totalLength - i;
                }
                return totalLength;
            }
        }

        if (seen.find(current) != seen.end()) {
            // We've found a loop; update cache for the chain
            int totalLength = chain.size();
            {
                lock_guard<mutex> lock(cache_mutex);
                for (size_t i = 0; i < chain.size(); ++i) {
                    cache[chain[i]] = totalLength - i;
                }
            }
            return totalLength;
        }

        chain.push_back(current);
        seen.insert(current);
        current = sumOfFactorials(current);
    }
}

// Thread function to compute chains in a range
void computeChains(int start, int end, unordered_map<int, int>& cache, atomic<int>& totalCount) {
    for (int i = start; i < end; ++i) {
        int chainLength = computeChain(i, cache);
        if (chainLength == 60) {
            totalCount++;
        }
    }
}

int main() {
    // Measure the execution time
    auto start_time = chrono::high_resolution_clock::now();

    atomic<int> totalCount(0);
    unordered_map<int, int> cache;

    // Determine the number of available hardware threads
    unsigned int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 4; // Default to 4 threads if unable to detect
    }

    vector<thread> threads;
    int maxNumber = 1000000;
    int range = maxNumber / numThreads;

    // Start threads to compute chains
    for (unsigned int i = 0; i < numThreads; ++i) {
        int start = i * range;
        int end = (i == numThreads - 1) ? maxNumber : start + range;
        threads.emplace_back(computeChains, start, end, ref(cache), ref(totalCount));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // Measure end time
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    // Show the total count and time taken
    cout << "Total chains with exactly 60 non-repeating terms: " << totalCount.load() << endl;
    cout << "Time taken: " << duration << " ms" << endl;

    return 0;
}