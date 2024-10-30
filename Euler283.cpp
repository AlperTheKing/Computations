#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <gmpxx.h> // Include GMP C++ header

using namespace std;

const int R_LMT = 20000;
mpz_class ans = 0;
int64_t total_triangles = 0; // Variable to count the number of triangles
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

        mpz_class local_ans = 0;
        int64_t local_count = 0; // Local count of triangles

        mpz_class r_squared = mpz_class(r) * r;

        for (mpz_class a = 1; a * a <= 3 * r_squared; ++a) {
            mpz_class temp = r_squared / a;
            mpz_class min_b = (temp > a) ? temp : a;
            for (mpz_class b = min_b; a * b <= 3 * r_squared; ++b) {
                mpz_class p = r_squared * (a + b);
                mpz_class q = a * b - r_squared;
                if (q <= 0)
                    continue;
                if (p < b * q)
                    break;
                if (mpz_divisible_p(p.get_mpz_t(), q.get_mpz_t())) {
                    mpz_class c = p / q;
                    mpz_class perimeter = 2 * (a + b + c);
                    local_ans += perimeter;
                    local_count++; // Increment local triangle count

                    // Uncomment the following lines to see the values of a+b, a+c, and b+c
                    /*
                    cout << "r = " << r << ": a+b = " << a + b
                         << ", a+c = " << a + c
                         << ", b+c = " << b + c << endl;
                    */
                }
            }
        }

        {
            lock_guard<mutex> lock(mtx);
            ans += local_ans;
            total_triangles += local_count; // Update global triangle count
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

    cout << "Total triangles found: " << total_triangles << endl;
    cout << "Total perimeter sum: " << ans.get_str() << endl;
    cout << "Execution time: " << duration << " milliseconds" << endl;

    return 0;
}