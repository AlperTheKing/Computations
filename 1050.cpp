#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

// Function to print a sequence of consecutive numbers starting from 'a' with length 'k'
void printSequence(long long a, int k) {
    std::cout << a;
    for(int i = 1; i < k; ++i){
        std::cout << " + " << (a + i);
    }
    std::cout << " = " << (a * k + (static_cast<long long>(k) * (k - 1)) / 2) << std::endl;
}

int main() {
    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    long long N = 1050;
    int count = 0;
    std::vector<std::pair<long long, int>> sequences; // To store valid sequences as (a, k)

    // Iterate over possible lengths of consecutive numbers (k)
    // The maximum possible k is when k*(k+1)/2 <= N
    for(int k = 1; k <= static_cast<int>((std::sqrt(2 * N + 0.25) - 0.5)); ++k){
        // Calculate the starting number 'a' of the consecutive sequence
        // Using the formula: a = (N/k) - (k-1)/2
        double a_double = static_cast<double>(N) / k - static_cast<double>(k - 1) / 2.0;
        long long a = static_cast<long long>(a_double);

        // 'a' must be a positive integer and the calculated 'a' should match 'a_double'
        if(a_double > 0 && std::floor(a_double) == a_double){
            // If k >= 2, it represents a sequence of at least two numbers
            if(k >= 2){
                count++;
                sequences.emplace_back(a, k);
            }
        }
    }

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results
    std::cout << "The number " << N << " can be expressed as the sum of consecutive integers in " 
              << count << " different ways.\n" << std::endl;

    std::cout << "Valid sequences:" << std::endl;
    for(const auto& seq : sequences){
        printSequence(seq.first, seq.second);
    }

    std::cout << "\nExecution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}