#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>

// Struct to hold the representation of a number as the sum of four squares
struct Representation {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t d;
};

// Function to find the representation of a number as the sum of up to four squares
Representation findRepresentation(int64_t n, const std::vector<int64_t>& squares) {
    Representation rep = {0, 0, 0, 0};
    
    // Check for one square
    int64_t a = static_cast<int64_t>(std::sqrt(n));
    if (a * a == n) {
        rep.a = a;
        return rep;
    }
    
    // Check for two squares
    for (a = 0; a <= static_cast<int64_t>(std::sqrt(n)); ++a) {
        int64_t remainder = n - a * a;
        int64_t b = static_cast<int64_t>(std::sqrt(remainder));
        if (b * b == remainder) {
            rep.a = a;
            rep.b = b;
            return rep;
        }
    }
    
    // Check for three squares
    for (a = 0; a <= static_cast<int64_t>(std::sqrt(n)); ++a) {
        int64_t remainder1 = n - a * a;
        for (int64_t b = 0; b <= static_cast<int64_t>(std::sqrt(remainder1)); ++b) {
            int64_t remainder2 = remainder1 - b * b;
            int64_t c = static_cast<int64_t>(std::sqrt(remainder2));
            if (c * c == remainder2) {
                rep.a = a;
                rep.b = b;
                rep.c = c;
                return rep;
            }
        }
    }
    
    // Use four squares
    for (a = 0; a <= static_cast<int64_t>(std::sqrt(n)); ++a) {
        int64_t remainder1 = n - a * a;
        for (int64_t b = 0; b <= static_cast<int64_t>(std::sqrt(remainder1)); ++b) {
            int64_t remainder2 = remainder1 - b * b;
            for (int64_t c = 0; c <= static_cast<int64_t>(std::sqrt(remainder2)); ++c) {
                int64_t d = static_cast<int64_t>(std::sqrt(n - a * a - b * b - c * c));
                if (d * d == (n - a * a - b * b - c * c)) {
                    rep.a = a;
                    rep.b = b;
                    rep.c = c;
                    rep.d = d;
                    return rep;
                }
            }
        }
    }
    
    // According to Lagrange's theorem, every number should have a representation
    return rep;
}

// Function executed by each thread to process a subset of numbers
void processRange(int64_t start, int64_t end, const std::vector<int64_t>& squares, std::vector<Representation>& representations) {
    for (int64_t n = start; n <= end; ++n) {
        representations[n] = findRepresentation(n, squares);
    }
}

int main() {
    const int64_t LOWER_BOUND = 1;
    const int64_t UPPER_BOUND = 1000000000; // 1b
    const unsigned int NUM_THREADS = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;

    std::cout << "Lagrange's Four Square Theorem Representation Finder\n";
    std::cout << "-----------------------------------------------------\n";
    std::cout << "Processing numbers from " << LOWER_BOUND << " to " << UPPER_BOUND << " using " << NUM_THREADS << " threads.\n";

    // Precompute all squares up to UPPER_BOUND
    std::vector<int64_t> squares;
    int64_t max_square_root = static_cast<int64_t>(std::sqrt(UPPER_BOUND));
    for (int64_t i = 0; i <= max_square_root; ++i) {
        squares.push_back(i * i);
    }

    // Preallocate representations vector (1-based indexing)
    std::vector<Representation> representations(UPPER_BOUND + 1);

    // Determine the range of numbers each thread will process
    std::vector<std::thread> threads;
    std::vector<std::pair<int64_t, int64_t>> ranges;
    int64_t numbers_per_thread = (UPPER_BOUND - LOWER_BOUND + 1) / NUM_THREADS;
    int64_t extra = (UPPER_BOUND - LOWER_BOUND + 1) % NUM_THREADS;
    int64_t current_start = LOWER_BOUND;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        int64_t current_end = current_start + numbers_per_thread - 1;
        if (i < extra) {
            current_end += 1;
        }
        if (current_end > UPPER_BOUND) {
            current_end = UPPER_BOUND;
        }
        ranges.emplace_back(std::make_pair(current_start, current_end));
        current_start = current_end + 1;
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(processRange, ranges[i].first, ranges[i].second, std::cref(squares), std::ref(representations));
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Processing completed in " << elapsed_seconds.count() << " seconds.\n";

    // Optional: Save representations to a file
    std::cout << "Saving representations to 'representations.txt'...\n";
    std::ofstream outfile("representations.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return 1;
    }

    for (int64_t n = LOWER_BOUND; n <= UPPER_BOUND; ++n) {
        const Representation& rep = representations[n];
        outfile << n << " = " << rep.a << "^2";
        if (rep.b > 0 || rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.b << "^2";
        }
        if (rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.c << "^2";
        }
        if (rep.d > 0) {
            outfile << " + " << rep.d << "^2";
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Representations saved successfully to 'representations.txt'.\n";

    // Optional: Display summary statistics
    // Count how many numbers require 1, 2, 3, or 4 squares
    int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
    for (int64_t n = LOWER_BOUND; n <= UPPER_BOUND; ++n) {
        int squares_used = 0;
        if (representations[n].a > 0) squares_used++;
        if (representations[n].b > 0) squares_used++;
        if (representations[n].c > 0) squares_used++;
        if (representations[n].d > 0) squares_used++;
        switch (squares_used) {
            case 1: count1++; break;
            case 2: count2++; break;
            case 3: count3++; break;
            case 4: count4++; break;
            default: break;
        }
    }

    std::cout << "\nSummary Statistics:\n";
    std::cout << "-------------------\n";
    std::cout << "Numbers expressed as the sum of 1 square: " << count1 << "\n";
    std::cout << "Numbers expressed as the sum of 2 squares: " << count2 << "\n";
    std::cout << "Numbers expressed as the sum of 3 squares: " << count3 << "\n";
    std::cout << "Numbers expressed as the sum of 4 squares: " << count4 << "\n";

    return 0;
}