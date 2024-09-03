#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <thread> // For std::this_thread::get_id

int main() {
    const int num_pieces = 40; // Number of pieces
    const long long num_simulations = 100000000000LL; // Number of simulations

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Combine thread ID and random_device for seeding
        std::random_device rd;
        unsigned long long seed = rd() ^ 
                                  (std::hash<std::thread::id>()(std::this_thread::get_id()) + 
                                  std::chrono::system_clock::now().time_since_epoch().count());

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(1, num_pieces);

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