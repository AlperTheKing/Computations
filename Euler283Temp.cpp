#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <cstdint>
using namespace std;

const int R_LMT = 1000;
int64_t global_max = 0;
int64_t ans = 0;
mutex mtx;
int r_current = 2;

void calculate() {
    while (true) {
        int r;
        {
            lock_guard<mutex> lock(mtx);
            if (r_current > 2 * R_LMT)
                break;
            r = r_current;
            r_current += 2;
        }

        int64_t local_ans = 0;
        int64_t local_max = 0;

        for (int64_t a = 1; a * a <= 3 * r * r; ++a) {
            for (int64_t b = max(r * r / a, a); a * b <= 3 * r * r; ++b) {
                int64_t p = r * r * (a + b);
                int64_t q = a * b - r * r;
                if (q <= 0)
                    continue;
                if (p < b * q)
                    break;
                if (p % q == 0) {
                    int64_t c = p / q;
                    int64_t sum_ab = a + b;
                    int64_t sum_ac = a + c;
                    int64_t sum_bc = b + c;

                    // Uncomment the following lines to see the values of a+b, a+c, b+c
                    // cout << "a+b: " << sum_ab << ", a+c: " << sum_ac << ", b+c: " << sum_bc << endl;

                    local_max = max({local_max, sum_ab, sum_ac, sum_bc});

                    int64_t perimeter = 2 * (a + b + c);
                    local_ans += perimeter;
                }
            }
        }

        {
            lock_guard<mutex> lock(mtx);
            ans += local_ans;
            global_max = max(global_max, local_max);
        }
    }
}

int main() {
    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;

    auto start_time = chrono::high_resolution_clock::now();

    // Launch worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(calculate);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    cout << "Largest value among all (a+b, a+c, b+c): " << global_max << endl;
    cout << "Total perimeter sum: " << ans << endl;
    cout << "Execution time: " << duration << " milliseconds" << endl;

    return 0;
}