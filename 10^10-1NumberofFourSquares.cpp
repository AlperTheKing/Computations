#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

// Function to find the representation of n as the sum of four squares
void findFourSquares(int64_t n, std::atomic<bool>& found, int64_t start, int64_t end, int64_t& a_out, int64_t& b_out, int64_t& c_out, int64_t& d_out) {
    int64_t max_value = static_cast<int64_t>(std::sqrt(n));
    for (int64_t a = start; a <= end && !found.load(); ++a) {
        int64_t a2 = a * a;
        if (a2 > n) break;

        for (int64_t b = a; b <= max_value && !found.load(); ++b) {
            int64_t b2 = b * b;
            if (a2 + b2 > n) break;

            for (int64_t c = b; c <= max_value && !found.load(); ++c) {
                int64_t c2 = c * c;
                if (a2 + b2 + c2 > n) break;

                int64_t d2 = n - a2 - b2 - c2;
                if (d2 < c2) continue;

                int64_t d = static_cast<int64_t>(std::sqrt(d2));
                if (d * d == d2) {
                    // Found a valid representation
                    a_out = a;
                    b_out = b;
                    c_out = c;
                    d_out = d;
                    found.store(true);
                    return;
                }
            }
        }
    }
}

int main() {
    int64_t n = 9999999999;

    // Number of threads to use
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    if (NUM_THREADS == 0) NUM_THREADS = 4; // Default to 4 if unable to detect

    std::cout << "Finding representation of " << n << " as the sum of four squares using " << NUM_THREADS << " threads.\n";

    // Variables to store the result
    std::atomic<bool> found(false);
    int64_t a_result = 0, b_result = 0, c_result = 0, d_result = 0;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the range of 'a' among threads
    int64_t max_value = static_cast<int64_t>(std::sqrt(n));
    int64_t range_per_thread = max_value / NUM_THREADS;
    int64_t extra = max_value % NUM_THREADS;

    std::vector<std::thread> threads;
    int64_t current_start = 0;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        int64_t current_end = current_start + range_per_thread;
        if (i < extra) current_end += 1;
        if (current_end > max_value) current_end = max_value;

        threads.emplace_back(findFourSquares, n, std::ref(found), current_start, current_end, std::ref(a_result), std::ref(b_result), std::ref(c_result), std::ref(d_result));

        current_start = current_end + 1;
    }

    // Wait for all threads to finish or until a result is found
    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Display the result
    if (found.load()) {
        std::cout << "Representation found:\n";
        std::cout << n << " = " << a_result << "^2 + " << b_result << "^2 + " << c_result << "^2 + " << d_result << "^2\n";
    } else {
        std::cout << "No representation found.\n";
    }

    std::cout << "Time taken: " << elapsed_seconds.count() << " seconds.\n";

    return 0;
}
