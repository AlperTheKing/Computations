#include <iostream>
#include <vector>
#include <unordered_set>
#include <thread>
#include <chrono>

// No need for mutexes since each thread writes to its own set
std::unordered_set<long long> set1, set2, set3;

void generate_set1(long long max_n) {
    for (long long k = 1; ; ++k) {
        long long n = 2 * k * k;
        if (n > max_n) break;
        set1.insert(n);
    }
}

void generate_set2(long long max_n) {
    for (long long m = 1; ; ++m) {
        long long n = 3 * m * m * m;
        if (n > max_n) break;
        set2.insert(n);
    }
}

void generate_set3(long long max_n) {
    for (long long p = 1; ; ++p) {
        // Compute n = 5 * p^5 directly
        long long n = 5 * p * p * p * p * p;
        if (n > max_n) break;
        set3.insert(n);
    }
}

int main() {
    const long long MAX_N = 10000000000000000LL; // Use LL suffix for long long literals
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create threads
    std::thread t1(generate_set1, MAX_N);
    std::thread t2(generate_set2, MAX_N);
    std::thread t3(generate_set3, MAX_N);

    // Wait for threads to finish
    t1.join();
    t2.join();
    t3.join();

    // Find intersection of the three sets
    std::vector<long long> intersection;
    for (const auto& n : set1) {
        if (set2.find(n) != set2.end() && set3.find(n) != set3.end()) {
            intersection.push_back(n);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Print results
    if (!intersection.empty()) {
        std::cout << "Found the following values of n satisfying all conditions:\n";
        for (const auto& n : intersection) {
            std::cout << n << "\n";
        }
    } else {
        std::cout << "No values of n found satisfying all conditions up to " << MAX_N << ".\n";
    }

    // Measure and print execution time
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Execution time: " << elapsed.count() << " seconds.\n";

    return 0;
}