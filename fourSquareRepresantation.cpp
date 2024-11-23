#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

// Function to check if n is expressible as the sum of two squares
bool isSumOfTwoSquares(int64_t n) {
    for (int64_t i = 2; i * i <= n; ++i) {
        int exponent = 0;
        if (n % i == 0) {
            while (n % i == 0) {
                n /= i;
                exponent++;
            }
            if (i % 4 == 3 && exponent % 2 != 0)
                return false;
        }
    }
    if (n > 1 && n % 4 == 3)
        return false;
    return true;
}

// Function to check if n is of the form 4^k(8m + 7)
bool isOfFormFourK8MPlus7(int64_t n) {
    while (n % 4 == 0) {
        n /= 4;
    }
    return n % 8 == 7;
}

// Function executed by each thread to process a subset of numbers
void processRange(int64_t start, int64_t end, std::atomic<int64_t>& count1, std::atomic<int64_t>& count2,
                  std::atomic<int64_t>& count3, std::atomic<int64_t>& count4, std::mutex& file_mutex, std::ofstream& outfile) {
    for (int64_t n = start; n <= end; ++n) {
        int squares_used = 4; // Default to four squares

        // Check for one square
        int64_t root = static_cast<int64_t>(std::sqrt(n));
        if (root * root == n) {
            squares_used = 1;

            // Optionally write the representation
            {
                std::lock_guard<std::mutex> guard(file_mutex);
                outfile << n << " = " << root << "^2\n";
            }
        }
        // Check for two squares
        else if (isSumOfTwoSquares(n)) {
            squares_used = 2;

            // Find the representation (a^2 + b^2 = n)
            int64_t limit = static_cast<int64_t>(std::sqrt(n / 2)) + 1;
            bool found = false;
            for (int64_t a = 1; a <= limit; ++a) {
                int64_t b2 = n - a * a;
                int64_t b = static_cast<int64_t>(std::sqrt(b2));
                if (b >= a && b * b == b2) { // Ensure b >= a to avoid duplicates
                    found = true;
                    {
                        std::lock_guard<std::mutex> guard(file_mutex);
                        outfile << n << " = " << a << "^2 + " << b << "^2\n";
                    }
                    break;
                }
            }
            if (!found) {
                // Should not happen due to the theorem, but include as a safety check
                {
                    std::lock_guard<std::mutex> guard(file_mutex);
                    outfile << n << " = sum of two squares (representation not found)\n";
                }
            }
        }
        // Check for three squares
        else if (!isOfFormFourK8MPlus7(n)) {
            squares_used = 3;

            // Find the representation (a^2 + b^2 + c^2 = n)
            int64_t limit = static_cast<int64_t>(std::sqrt(n / 3)) + 1;
            bool found = false;
            for (int64_t a = 0; a <= limit && !found; ++a) {
                int64_t remainder1 = n - a * a;
                int64_t limit_b = static_cast<int64_t>(std::sqrt(remainder1 / 2)) + 1;
                for (int64_t b = a; b <= limit_b && !found; ++b) {
                    int64_t c2 = remainder1 - b * b;
                    int64_t c = static_cast<int64_t>(std::sqrt(c2));
                    if (c >= b && c * c == c2) { // Ensure c >= b to avoid duplicates
                        found = true;
                        {
                            std::lock_guard<std::mutex> guard(file_mutex);
                            outfile << n << " = " << a << "^2 + " << b << "^2 + " << c << "^2\n";
                        }
                        break;
                    }
                }
            }
            if (!found) {
                // Should not happen due to the theorem, but include as a safety check
                {
                    std::lock_guard<std::mutex> guard(file_mutex);
                    outfile << n << " = sum of three squares (representation not found)\n";
                }
            }
        }
        // Four squares
        else {
            squares_used = 4;

            // Optionally write a default message
            {
                std::lock_guard<std::mutex> guard(file_mutex);
                outfile << n << " = sum of four squares\n";
            }
        }

        // Update counts
        if (squares_used == 1) count1++;
        else if (squares_used == 2) count2++;
        else if (squares_used == 3) count3++;
        else count4++;
    }
}

int main() {
    const int64_t LOWER_BOUND = 1;
    const int64_t UPPER_BOUND = 1000000000; // 1e9
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    if (NUM_THREADS == 0) NUM_THREADS = 4; // Default to 4 threads if hardware_concurrency() returns 0

    std::cout << "Lagrange's Four Square Theorem Representation Finder\n";
    std::cout << "-----------------------------------------------------\n";
    std::cout << "Processing numbers from " << LOWER_BOUND << " to " << UPPER_BOUND << " using " << NUM_THREADS << " threads.\n";

    // Atomic counters for thread-safe updates
    std::atomic<int64_t> count1(0), count2(0), count3(0), count4(0);

    // Open output file
    std::ofstream outfile("representations.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return 1;
    }

    // Mutex for thread-safe file writing
    std::mutex file_mutex;

    // Divide the range among threads
    std::vector<std::thread> threads;
    int64_t total_numbers = UPPER_BOUND - LOWER_BOUND + 1;
    int64_t numbers_per_thread = total_numbers / NUM_THREADS;
    int64_t extra = total_numbers % NUM_THREADS;

    int64_t current_start = LOWER_BOUND;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        int64_t current_end = current_start + numbers_per_thread - 1;
        if (i < extra) {
            current_end += 1;
        }
        if (current_end > UPPER_BOUND) {
            current_end = UPPER_BOUND;
        }

        threads.emplace_back(processRange, current_start, current_end, std::ref(count1), std::ref(count2),
                             std::ref(count3), std::ref(count4), std::ref(file_mutex), std::ref(outfile));

        current_start = current_end + 1;
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Processing completed in " << elapsed_seconds.count() << " seconds.\n";

    // Close the output file
    outfile.close();
    std::cout << "Representations saved successfully to 'representations.txt'.\n";

    // Output summary statistics
    std::cout << "\nSummary Statistics:\n";
    std::cout << "-------------------\n";
    std::cout << "Numbers expressed as the sum of 1 square: " << count1.load() << "\n";
    std::cout << "Numbers expressed as the sum of 2 squares: " << count2.load() << "\n";
    std::cout << "Numbers expressed as the sum of 3 squares: " << count3.load() << "\n";
    std::cout << "Numbers expressed as the sum of 4 squares: " << count4.load() << "\n";

    return 0;
}
