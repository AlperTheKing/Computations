#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <thread>
#include "Random123/philox.h"  // Include the Random123 Philox header

using namespace r123;  // Use the Random123 namespace

int main() {
    const int num_pieces = 40; // Number of pieces
    const long long num_simulations = 100000000000LL; // Number of simulations

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Get the thread ID to use as part of the counter for Random123
        unsigned int tid = omp_get_thread_num();

        long long local_total = 0;

        std::vector<int> pieces(num_pieces);
        std::vector<bool> placed(num_pieces + 1, false);

        #pragma omp for
        for (long long i = 0; i < num_simulations; ++i) {
            std::iota(pieces.begin(), pieces.end(), 1);

            // Initialize Philox RNG from Random123
            Philox4x32 rng;  // Instantiate the RNG
            Philox4x32::ctr_type c = {{}};
            Philox4x32::key_type k = {{}};

            k[0] = tid;  // Use thread ID as part of the key
            k[1] = i;    // Use iteration number as another part of the key

            // Fisher-Yates shuffle using Philox RNG
            for (int j = num_pieces - 1; j > 0; --j) {
                c[0] = j;
                c[1] = i;
                Philox4x32::ctr_type r = rng(c, k);  // Generate random numbers
                int rand_index = r[0] % (j + 1);
                std::swap(pieces[j], pieces[rand_index]);
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