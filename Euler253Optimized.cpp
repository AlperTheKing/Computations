#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <./xoshiro256plus.h> // Assuming you have a Xoshiro256+ header, or you can implement it yourself

int main() {
    const int num_pieces = 40; // Number of pieces
    const long long num_simulations = 100000000000LL; // Number of simulations

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Use std::random_device to seed Xoshiro256+ RNG
        std::random_device rd;
        xoshiro256plus rng(rd(), rd(), rd(), rd());

        long long local_total = 0;

        std::vector<int> pieces(num_pieces);
        std::vector<bool> placed(num_pieces + 1, false); // Allocate once per thread

        #pragma omp for
        for (long long i = 0; i < num_simulations; ++i) {
            std::iota(pieces.begin(), pieces.end(), 1);

            // Optimized Fisher-Yates shuffle using Xoshiro256+ RNG
            for (int j = num_pieces - 1; j > 0; --j) {
                int r = rng() % (j + 1); // Optimized random index generation
                std::swap(pieces[j], pieces[r]);
            }

            std::fill(placed.begin(), placed.end(), false);
            int segments = 0;
            int max_segments = 0;

            for (int j = 0; j < num_pieces; ++j) {
                int piece = pieces[j];
                placed[piece] = true;

                if (placed[piece - 1] && placed[piece + 1]) {
                    segments--;
                } else if (!placed[piece - 1] && !placed[piece + 1]) {
                    segments++;
                }

                max_segments = std::max(max_segments, segments);
            }

            local_total += max_segments;
        }

        #pragma omp atomic
        total_max_segments += local_total;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double average_max_segments = static_cast<double>(total_max_segments) / num_simulations;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average maximum number of segments: " << average_max_segments << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}