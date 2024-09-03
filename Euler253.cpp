#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iomanip>

int main() {
    const int num_pieces = 40; // Number of pieces
    const long long num_simulations = 10000000000LL; // Number of simulations

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::mt19937 rng(std::random_device{}() + omp_get_thread_num()); // Unique seed per thread
        long long local_total = 0;

        #pragma omp for
        for (long long i = 0; i < num_simulations; ++i) {
            std::vector<int> pieces(num_pieces);
            std::iota(pieces.begin(), pieces.end(), 1);
            std::shuffle(pieces.begin(), pieces.end(), rng);

            std::vector<bool> placed(num_pieces + 1, false);
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