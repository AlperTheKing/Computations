#include <iostream>
#include <vector>
#include <algorithm>
#include <random>  // For random number generation
#include <numeric> // For std::iota
#include <omp.h>
#include <chrono>
#include <iomanip> // For setting precision

int main() {
    const int num_pieces = 40; // Changed to 10 pieces
    const int num_simulations = 1000000000; // Number of simulations for averaging

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::mt19937 rng(std::random_device{}()); // Thread-local random number generator
        long long local_total = 0;

        #pragma omp for
        for (int i = 0; i < num_simulations; ++i) {
            std::vector<int> pieces(num_pieces);
            std::iota(pieces.begin(), pieces.end(), 1); // Fill with 1 to num_pieces
            std::shuffle(pieces.begin(), pieces.end(), rng); // Randomly shuffle pieces

            std::vector<bool> placed(num_pieces + 1, false); // Track placed pieces
            int segments = 0;
            int max_segments = 0;

            for (int j = 0; j < num_pieces; ++j) {
                int piece = pieces[j];
                placed[piece] = true;

                // Check if piece creates or merges segments
                if (placed[piece - 1] && placed[piece + 1]) {
                    segments--; // Merges two segments
                } else if (!placed[piece - 1] && !placed[piece + 1]) {
                    segments++; // Creates a new segment
                }
                
                max_segments = std::max(max_segments, segments); // Update max segments
            }

            local_total += max_segments;
        }

        #pragma omp atomic
        total_max_segments += local_total;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double average_max_segments = static_cast<double>(total_max_segments) / num_simulations;

    // Set precision to 6 decimal places
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average maximum number of segments: " << average_max_segments << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}