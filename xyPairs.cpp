#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <cmath> // Include cmath for mathematical functions

// Optionally bring ceil and floor into scope
using std::ceil;
using std::floor;

std::mutex mtx; // Mutex for synchronizing access to shared resources

// Global vector to store all pairs
std::vector<std::pair<int, int>> allPairs;

void countAndCollectValidPairs(int startX, int endX, int& totalPairs) {
    std::vector<std::pair<int, int>> localPairs; // Local storage for pairs

    for (int x = startX; x <= endX; ++x) {
        double minY = std::max(x / 2.0, 1.0);       // Corrected calculation
        double maxY = std::min(2.0 * x, 60.0);      // y <= 2x and 2x <= 60
        maxY = std::min(maxY, 30.0);                // y <= 30

        int yStart = static_cast<int>(ceil(minY));
        int yEnd = static_cast<int>(floor(maxY));

        for (int y = yStart; y <= yEnd; ++y) {
            if (2 * y >= x && y <= 2 * x && 2 * y <= 60 && 2 * x <= 60) {
                localPairs.emplace_back(x, y);
            }
        }
    }

    // Add local pairs to the global vector
    {
        std::lock_guard<std::mutex> lock(mtx);
        totalPairs += localPairs.size();
        allPairs.insert(allPairs.end(), localPairs.begin(), localPairs.end());
    }
}

int main() {
    int totalPairs = 0;
    int maxX = 30; // Maximum value of x

    // Determine the number of hardware threads available
    unsigned int threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 4; // Default to 4 threads if unable to detect

    // Time measurement start
    auto startTime = std::chrono::high_resolution_clock::now();

    // Calculate the range of x values for each thread
    int rangePerThread = maxX / threadCount;
    int remainder = maxX % threadCount;

    std::vector<std::thread> threads;

    int currentX = 1;
    for (unsigned int i = 0; i < threadCount; ++i) {
        int startX = currentX;
        int endX = currentX + rangePerThread - 1;
        if (i < remainder) {
            endX++; // Distribute the remainder among the first few threads
        }
        currentX = endX + 1;

        threads.emplace_back(countAndCollectValidPairs, startX, endX, std::ref(totalPairs));
    }

    // Join all threads
    for (auto& th : threads) {
        th.join();
    }

    // Sort the global vector
    std::sort(allPairs.begin(), allPairs.end());

    // Time measurement end
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Print the sorted pairs
    for (const auto& pair : allPairs) {
        std::cout << "x = " << pair.first << ", y = " << pair.second << std::endl;
    }

    // Output the result and execution time
    std::cout << "Total valid pairs: " << totalPairs << std::endl;
    std::cout << "Execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    return 0;
}