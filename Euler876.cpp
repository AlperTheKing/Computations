#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>
#include <queue>
#include <set>
#include <functional>
#include <atomic>
#include <tuple>
#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace std::chrono;


using namespace boost::multiprecision;

// Integer power function
int128_t int_pow(int128_t base, int exp) {
    int128_t result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// Function to compute f(a, b, c) with BFS
int compute_f(int128_t a, int128_t b, int128_t c, int max_depth) {
    int128_t sum = a + b + c;
    if (sum % 2 != 0)
        return 0; // m is not integer

    int128_t m = sum / 2;
    int128_t delta_a0 = a - m;
    int128_t delta_b0 = b - m;
    int128_t delta_c0 = c - m;

    // Check if sum of deviations is -m
    if ((delta_a0 + delta_b0 + delta_c0) != -m)
        return 0;

    // Initialize BFS
    struct State {
        int steps;
        int128_t delta_a;
        int128_t delta_b;
        int128_t delta_c;
    };

    queue<State> q;
    q.push({0, delta_a0, delta_b0, delta_c0});
    set<tuple<int128_t, int128_t, int128_t>> visited;

    while (!q.empty()) {
        State curr = q.front();
        q.pop();

        // Create a unique key for the current state
        auto state_key = make_tuple(curr.delta_a, curr.delta_b, curr.delta_c);

        // Check if current state is visited
        if (visited.find(state_key) != visited.end())
            continue;
        visited.insert(state_key);

        // Check if any variable is zero (delta_i == -m)
        if (curr.delta_a == -m || curr.delta_b == -m || curr.delta_c == -m)
            return curr.steps;

        // Limit the depth of BFS to avoid long runtimes
        if (curr.steps >= max_depth)
            continue;

        // Generate new states by applying operation on each variable
        vector<int128_t*> deltas = { &curr.delta_a, &curr.delta_b, &curr.delta_c };

        for (auto delta_ptr : deltas) {
            int128_t original_delta = *delta_ptr;
            *delta_ptr = -3 * (*delta_ptr);
            q.push({curr.steps + 1, curr.delta_a, curr.delta_b, curr.delta_c});
            *delta_ptr = original_delta; // Revert change
        }
    }

    return 0; // No solution found
}

// Worker function for processing c values
void c_worker(int128_t a, int128_t b, atomic<int64_t>& next_c, int128_t c_max, mutex& F_ab_mutex, int128_t& F_ab, int max_depth) {
    while (true) {
        int64_t c = next_c.fetch_add(1);
        if (c > c_max)
            break;

        int steps = compute_f(a, b, c, max_depth);
        if (steps > 0) {
            lock_guard<mutex> guard(F_ab_mutex);
            F_ab += steps;
        }
    }
}

int main() {
    auto start_time = high_resolution_clock::now();

    const int num_threads = thread::hardware_concurrency();
    const int128_t c_max = 1000; // Adjust c_max as needed
    const int max_depth = 500; // Increase BFS depth limit to 1000

    vector<int128_t> F_values(18, 0);

    for (int k = 1; k <= 18; ++k) {
        int128_t a = int_pow(6, k);
        int128_t b = int_pow(10, k);
        int128_t F_ab = 0;
        atomic<int64_t> next_c(1);
        vector<thread> c_threads;
        mutex F_ab_mutex;

        // Launch threads for c loop
        for (int i = 0; i < num_threads; ++i) {
            c_threads.emplace_back(c_worker, a, b, ref(next_c), c_max, ref(F_ab_mutex), ref(F_ab), max_depth);
        }

        // Wait for all threads to finish
        for (auto& th : c_threads) {
            th.join();
        }

        F_values[k - 1] = F_ab;
        cout << "Computed F(" << a << ", " << b << ") = " << F_values[k - 1] << endl;
    }

    // Compute total sum S
    int128_t total_S = 0;
    for (int i = 0; i < 18; ++i) {
        total_S += F_values[i];
    }

    cout << "Total S = " << total_S << endl;

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);

    cout << "Execution Time: " << duration.count() << " seconds" << endl;

    return 0;
}