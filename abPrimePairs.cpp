#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>
#include <queue>
#include <condition_variable>

// Mutexes and condition variables for thread synchronization
std::mutex mtx_output;
std::mutex mtx_queue;
std::condition_variable cv;

// Shared task queue
std::queue<uint64_t> taskQueue;
bool done = false;

// Function to check if a number is prime (optimized basic method)
bool isPrime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

// Function to compute integer cube root
uint64_t integerCubeRoot(uint64_t x) {
    uint64_t lo = 0, hi = std::cbrt(x) + 1;
    while (lo <= hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        __uint128_t midCubed = (__uint128_t)mid * mid * mid;
        if (midCubed == x)
            return mid;
        else if (midCubed < x)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return 0;
}

// Worker function for each thread
void workerFunction(uint64_t& count) {
    while (true) {
        uint64_t a;

        // Retrieve a task from the queue
        {
            std::unique_lock<std::mutex> lock(mtx_queue);
            cv.wait(lock, [] { return !taskQueue.empty() || done; });

            if (taskQueue.empty()) {
                if (done) {
                    return; // No more tasks and done flag is true
                } else {
                    continue; // Spurious wakeup or tasks not yet added
                }
            }

            a = taskQueue.front();
            taskQueue.pop();
        }

        // Process the task outside the lock
        if (!isPrime(a)) continue;

        __uint128_t aCubed = (__uint128_t)a * a * a;
        __uint128_t s = 20 * aCubed - 1;
        uint64_t b = integerCubeRoot(s);
        if (b != 0 && (__uint128_t)b * b * b == s && isPrime(b)) {
            std::lock_guard<std::mutex> lock(mtx_output);
            std::cout << "Found pair: a = " << a << ", b = " << b << std::endl;
            count++;
        }
    }
}

int main() {
    uint64_t maxA = 1000000000; // Adjust this range as needed (up to 10^10)
    uint64_t threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 4; // Default to 4 threads if hardware_concurrency returns 0
    uint64_t count = 0;

    // Time measurement start
    auto startTime = std::chrono::high_resolution_clock::now();

    // Populate the task queue before starting worker threads
    {
        std::lock_guard<std::mutex> lock(mtx_queue);
        for (uint64_t a = 2; a <= maxA; a++) { // Include all numbers; primality is checked later
            taskQueue.push(a);
        }
    }

    // Start worker threads
    std::vector<std::thread> threads;
    for (uint64_t i = 0; i < threadCount; ++i) {
        threads.emplace_back(workerFunction, std::ref(count));
    }

    // Notify threads that tasks are available
    cv.notify_all();

    // After all tasks are added, set done to true
    {
        std::lock_guard<std::mutex> lock(mtx_queue);
        done = true;
    }
    cv.notify_all(); // Wake up any threads waiting on the condition variable

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Time measurement end
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout << "Total pairs found: " << count << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}