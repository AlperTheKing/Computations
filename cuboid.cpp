#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <queue>
#include <condition_variable>

// Maximum side length (set to 1,000,000 as per requirement)
// WARNING: This value leads to an extremely large number of iterations.
// It is highly recommended to use a smaller value for practical execution.
const int64_t MAX_SIDE = 10000000;

// Structure to store brick dimensions
struct Brick {
    int64_t a;
    int64_t b;
    int64_t c;
};

// Function to check if a number is a perfect square using floating-point operations
inline bool isPerfectSquare(int64_t x) {
    if (x < 0) return false;
    double root = std::sqrt(static_cast<double>(x));
    int64_t root_int = static_cast<int64_t>(root + 1e-9); // Adding a small epsilon to handle floating-point precision
    return root_int * root_int == x;
}

// Task queue and synchronization primitives
std::queue<std::pair<int64_t, int64_t>> taskQueue;
std::mutex queueMutex;
std::condition_variable cv;
bool done = false;

// Function to retrieve a task from the queue
bool getTask(std::pair<int64_t, int64_t> &task) {
    std::unique_lock<std::mutex> lock(queueMutex);
    while (taskQueue.empty() && !done) {
        cv.wait(lock);
    }
    if (!taskQueue.empty()) {
        task = taskQueue.front();
        taskQueue.pop();
        return true;
    }
    return false;
}

// Worker function to process a range of 'a' values
void workerFunction(std::vector<Brick> &localBricks) {
    std::pair<int64_t, int64_t> task;
    while (getTask(task)) {
        int64_t start_a = task.first;
        int64_t end_a = task.second;

        for (int64_t a = start_a; a <= end_a; ++a) {
            int64_t a_sq = a * a;
            for (int64_t b = a; b <= MAX_SIDE; ++b) { // Ensure b >= a
                int64_t b_sq = b * b;
                int64_t diag_ab_sq = a_sq + b_sq;
                if (!isPerfectSquare(diag_ab_sq)) continue;

                for (int64_t c = b; c <= MAX_SIDE; ++c) { // Ensure c >= b
                    int64_t c_sq = c * c;
                    int64_t diag_ac_sq = a_sq + c_sq;
                    if (!isPerfectSquare(diag_ac_sq)) continue;

                    int64_t diag_bc_sq = b_sq + c_sq;
                    if (!isPerfectSquare(diag_bc_sq)) continue;

                    int64_t diag_abc_sq = a_sq + b_sq + c_sq;
                    if (!isPerfectSquare(diag_abc_sq)) continue;

                    // All conditions met, add to local list
                    Brick brick = {a, b, c};
                    localBricks.push_back(brick);
                }
            }
        }
    }
}

// Function to populate the task queue with ranges of 'a' values
void populateTasks(int64_t chunkSize) {
    for (int64_t start = 1; start <= MAX_SIDE; start += chunkSize) {
        int64_t end = std::min(start + chunkSize - 1, MAX_SIDE);
        taskQueue.emplace(start, end);
    }
}

int main() {
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine the number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default to 4 if unable to detect
    std::cout << "Number of threads used: " << num_threads << "\n";

    // Define the size of each task (number of 'a' values per task)
    const int64_t TASK_CHUNK_SIZE = 100; // Adjust based on performance/memory

    // Populate the task queue
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        populateTasks(TASK_CHUNK_SIZE);
        done = true; // All tasks have been enqueued
    }
    cv.notify_all(); // Notify all waiting threads

    // Create a vector to hold bricks found by all threads
    std::vector<Brick> allBricks;
    // No mutex needed here as we'll use local accumulation

    // Launch worker threads
    std::vector<std::thread> threads;
    // Each thread will have its local bricks to minimize locking
    std::vector<std::vector<Brick>> threadLocalBricks(num_threads, std::vector<Brick>());

    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(workerFunction, std::ref(threadLocalBricks[i]));
    }

    // Wait for all threads to finish
    for (auto &th : threads) {
        th.join();
    }

    // Merge all local bricks into the main list
    for (const auto &localBricks : threadLocalBricks) {
        allBricks.insert(allBricks.end(), localBricks.begin(), localBricks.end());
    }

    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output the results
    std::cout << "Number of bricks found: " << allBricks.size() << "\n";

    // Uncomment the following lines if you wish to print all bricks
    /*
    for (const auto &brick : allBricks) {
        std::cout << "a: " << brick.a << ", b: " << brick.b << ", c: " << brick.c << "\n";
    }
    */

    // Output the total elapsed time
    std::cout << "Total time: " << elapsed.count() << " seconds\n";

    return 0;
}