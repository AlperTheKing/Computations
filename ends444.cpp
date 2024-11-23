#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdint>   // For int64_t
#include <algorithm> // For std::sort

// Function to check numbers in a subrange and store results
void findNumbersEndingWith444(int64_t start, int64_t end, std::vector<int64_t>& results, std::mutex& mtx) {
    std::vector<int64_t> localResults;
    for (int64_t n = start; n <= end; ++n) {
        __int128 square = static_cast<__int128>(n) * n; // Use __int128 to prevent overflow
        if (square % 1000 == 444) {
            localResults.push_back(n);
        }
    }
    // Lock and add local results to the shared results vector
    std::lock_guard<std::mutex> lock(mtx);
    results.insert(results.end(), localResults.begin(), localResults.end());
}

int main() {
    // Define the search range
    int64_t lowerBound = 1;
    int64_t upperBound = 1000000000; // You can adjust this range as needed

    // Determine the number of hardware threads available
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 2; // Fallback if unable to detect
    std::cout << "Using " << numThreads << " threads.\n";

    // Calculate the size of each subrange
    int64_t rangeSize = (upperBound - lowerBound + 1) / numThreads;

    // Vector to hold the results
    std::vector<int64_t> results;
    // Mutex to protect shared resources
    std::mutex mtx;
    // Vector to hold thread objects
    std::vector<std::thread> threads;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch threads to process subranges
    for (unsigned int i = 0; i < numThreads; ++i) {
        int64_t start = lowerBound + i * rangeSize;
        int64_t end = (i == numThreads - 1) ? upperBound : (start + rangeSize - 1);
        threads.emplace_back(findNumbersEndingWith444, start, end, std::ref(results), std::ref(mtx));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    // Sort the results
    std::sort(results.begin(), results.end());

    // Output the results
    std::cout << "Numbers whose square ends with 444:\n";
    for (const auto& num : results) {
        __int128 square = static_cast<__int128>(num) * num;
        // To print __int128, we need to convert it to a string
        std::string squareStr;
        __int128 temp = square;
        bool isNegative = false;
        if (temp < 0) {
            isNegative = true;
            temp = -temp;
        }
        do {
            squareStr.insert(squareStr.begin(), '0' + (temp % 10));
            temp /= 10;
        } while (temp > 0);
        if (isNegative) {
            squareStr.insert(squareStr.begin(), '-');
        }
        std::cout << num << " (Square: " << squareStr << ")\n";
    }

    // Output the time taken
    std::cout << "Total numbers found: " << results.size() << "\n";
    std::cout << "Time taken: " << elapsed.count() << " seconds.\n";

    return 0;
}