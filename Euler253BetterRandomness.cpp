#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <thread> // For std::this_thread::get_id

// Xoshiro256+ RNG class (can replace this with a proper implementation or include a library)
class xoshiro256plus {
public:
    xoshiro256plus(uint64_t seed1, uint64_t seed2, uint64_t seed3, uint64_t seed4) {
        s[0] = seed1;
        s[1] = seed2;
        s[2] = seed3;
        s[3] = seed4;
    }

    uint64_t operator()() {
        const uint64_t result = s[0] + s[3];
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

private:
    uint64_t s[4];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

int main() {
    const int num_pieces = 40; // Number of pieces
    const long long num_simulations = 100000000000LL; // Number of simulations

    long long total_max_segments = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Combine random_device and thread ID for seeding
        std::random_device rd;
        uint64_t seed1 = rd();
        uint64_t seed2 = rd();
        uint64_t seed3 = rd();
        uint64_t seed4 = rd();
        uint64_t thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());

        xoshiro256plus rng(seed1 ^ thread_id, seed2 ^ thread_id, seed3 ^ thread_id, seed4 ^ thread_id);

        long long local_total = 0;

        std::vector<int> pieces(num_pieces);
        std::vector<bool> placed(num_pieces + 1, false);

        #pragma omp for
        for (long long i = 0; i < num_simulations; ++i) {
            std::iota(pieces.begin(), pieces.end(), 1);

            // Fisher-Yates shuffle using Xoshiro256+ RNG
            for (int j = num_pieces - 1; j > 0; --j) {
                int r = rng() % (j + 1);
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