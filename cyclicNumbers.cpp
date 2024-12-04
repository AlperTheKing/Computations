#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <cmath>
#include <algorithm>
#include <string>

// Function to generate the cyclic number for a given prime p
std::string generateCyclicNumber(int p) {
    std::string cyclicNumber;
    int remainder = 1;
    for (int i = 0; i < p - 1; ++i) {
        int digit = (remainder * 10) / p;
        cyclicNumber += std::to_string(digit);
        remainder = (remainder * 10) % p;
    }
    return cyclicNumber;
}

// Function to perform string multiplication: result = number * multiplier
std::string multiplyStringByInt(const std::string& number, int multiplier) {
    std::string result;
    int carry = 0;
    for (int i = number.size() - 1; i >= 0; --i) {
        int digit = number[i] - '0';
        int product = digit * multiplier + carry;
        result.push_back('0' + (product % 10));
        carry = product / 10;
    }
    while (carry > 0) {
        result.push_back('0' + (carry % 10));
        carry /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// Function to check if a number is cyclic
bool isCyclicNumber(const std::string& number) {
    int n = number.length();

    for (int i = 2; i <= n; ++i) {
        std::string multipleStr = multiplyStringByInt(number, i);

        // If the length doesn't match, it's not a cyclic number
        if (multipleStr.length() != n) {
            return false;
        }

        // Check if multipleStr is a cyclic permutation of number
        std::string temp = number + number; // Concatenate to handle wrap-around
        if (temp.find(multipleStr) == std::string::npos) {
            return false;
        }
    }
    return true;
}

// Convert string to __int128_t
bool stringToInt128(const std::string& str, __int128_t& value) {
    value = 0;
    for (char c : str) {
        if (c < '0' || c > '9') return false;
        value = value * 10 + (c - '0');
        if (value < 0) return false; // Overflow detection
    }
    return true;
}

// Convert __int128_t to string
std::string int128ToString(__int128_t value) {
    if (value == 0) return "0";
    std::string result;
    while (value > 0) {
        int digit = value % 10;
        result.push_back('0' + digit);
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// Task structure for dynamic load balancing
struct Task {
    int prime;
};

// Thread-safe queue
class SafeQueue {
private:
    std::queue<Task> tasks;
    std::mutex mtx;
public:
    void enqueue(const Task& task) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(task);
    }

    bool dequeue(Task& task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) {
            return false;
        }
        task = tasks.front();
        tasks.pop();
        return true;
    }
};

// Worker function for threads
void worker(SafeQueue& taskQueue, std::mutex& outputMtx, std::string& maxCyclicNumber, int& maxPrime) {
    Task task;
    while (taskQueue.dequeue(task)) {
        int p = task.prime;
        std::string cyclicStr = generateCyclicNumber(p);

        // Convert cyclicStr to __int128_t
        __int128_t cyclicNum;
        if (!stringToInt128(cyclicStr, cyclicNum)) {
            // Overflow occurred, skip this prime
            continue;
        }

        // Check if cyclicNum fits within __int128_t
        if (cyclicNum > 0) {
            // Check if it's a cyclic number
            if (isCyclicNumber(cyclicStr)) {
                std::lock_guard<std::mutex> lock(outputMtx);

                // Compare lengths first, then lexicographically
                if (cyclicStr.length() > maxCyclicNumber.length() ||
                    (cyclicStr.length() == maxCyclicNumber.length() && cyclicStr > maxCyclicNumber)) {
                    maxCyclicNumber = cyclicStr;
                    maxPrime = p;
                    std::cout << "Found cyclic number for prime " << p << ": " << cyclicStr << "\n";
                }
            }
        }
    }
}

int main() {
    // List of primes to check (Full Reptend Primes in base 10)
    std::vector<int> primes = {
        7, 17, 19, 23, 29, 47, 59, 61, 97, 109,
        113, 131, 149, 157, 167, 179, 181, 193, 223, 229,
        233, 239, 241, 251, 257, 263, 269, 283, 307, 311,
        313, 317, 331, 337, 353, 359, 367, 379, 383, 389,
        397, 409, 419, 421, 433, 439, 443, 449, 457, 461,
        463, 487, 491, 499, 503, 509, 521, 523, 541
    };

    // Remove primes where p - 1 digits exceed 39 (max digits in __int128_t)
    primes.erase(std::remove_if(primes.begin(), primes.end(),
                [](int p) { return p - 1 > 39; }),
                primes.end());

    // Start time measurement
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize task queue
    SafeQueue taskQueue;
    for (int p : primes) {
        taskQueue.enqueue({p});
    }

    // Output mutex
    std::mutex outputMtx;

    // Variables to store the maximum cyclic number found
    std::string maxCyclicNumber = "";
    int maxPrime = 0;

    // Determine the number of threads
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 if unable to detect

    std::cout << "Using " << numThreads << " threads.\n";

    // Create and launch threads
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, std::ref(taskQueue), std::ref(outputMtx),
                             std::ref(maxCyclicNumber), std::ref(maxPrime));
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // End time measurement
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Output the result
    if (!maxCyclicNumber.empty()) {
        std::cout << "\nLargest cyclic number within __int128_t is for prime " << maxPrime << ":\n";
        std::cout << maxCyclicNumber << "\n";

        // Show the multiplication table
        std::cout << "\nDemonstrating the cyclic property:\n";
        int n = maxCyclicNumber.length();
        for (int i = 1; i <= n; ++i) {
            std::string multipleStr = multiplyStringByInt(maxCyclicNumber, i);

            // Ensure the multiple has leading zeros if necessary
            while (multipleStr.length() < n) {
                multipleStr = "0" + multipleStr;
            }

            std::cout << maxCyclicNumber << " x " << i << " = " << multipleStr << "\n";
        }
    } else {
        std::cout << "\nNo cyclic number found within __int128_t range.\n";
    }

    std::cout << "Time taken: " << elapsedSeconds.count() << " seconds.\n";

    return 0;
}
