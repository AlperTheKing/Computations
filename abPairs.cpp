#include <iostream>
#include <cmath>
#include <gmp.h>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

const uint64_t A_MAX = 9000;
const uint64_t B_MAX = 9000;
std::mutex count_mutex;
uint64_t pair_count = 0;
vector<pair<uint64_t, uint64_t>> pairs;

void findPairs(uint64_t start, uint64_t end) {
    uint64_t local_count = 0;
    mpz_t sum, term1, term2, sqrt_sum;
    mpz_init(sum);
    mpz_init(term1);
    mpz_init(term2);
    mpz_init(sqrt_sum);
    vector<pair<uint64_t, uint64_t>> local_pairs;
    for (uint64_t a = start; a < end; ++a) {
        for (uint64_t b = 0; b <= B_MAX; ++b) {
            mpz_t sum, term1, term2;
            mpz_init(sum);
            mpz_init(term1);
            mpz_init(term2);
            mpz_ui_pow_ui(term1, 2, 2 * a);
            mpz_ui_pow_ui(term2, 2, b);
            mpz_add(sum, term1, term2);
            mpz_t sqrt_sum;
            mpz_init(sqrt_sum);
            mpz_sqrt(sqrt_sum, sum);
            if (mpz_perfect_square_p(sum)) {
                ++local_count;
                local_pairs.emplace_back(a, b);
                mpz_clear(sum);
    mpz_clear(term1);
    mpz_clear(term2);
    mpz_clear(sqrt_sum);
}
        }
    }
    std::lock_guard<std::mutex> lock(count_mutex);
    pair_count += local_count;
    pairs.insert(pairs.end(), local_pairs.begin(), local_pairs.end());
}

int main() {
    auto start_time = chrono::high_resolution_clock::now();
    
    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;
    uint64_t range = A_MAX / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        uint64_t start = i * range;
        uint64_t end = (i == num_threads - 1) ? A_MAX : (i + 1) * range;
        threads.emplace_back(findPairs, start, end);
    }

    for (auto &t : threads) {
        t.join();
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    for (const auto &p : pairs) {
        cout << "(a, b) = (" << p.first << ", " << p.second << ")" << endl;
    }
    cout << "Number of (a, b) pairs: " << pair_count << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}