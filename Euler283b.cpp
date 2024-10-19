#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>

// Use __int128 for 128-bit integer support
typedef __int128 int128_t;

using namespace std;

const int R_LMT = 9000;
int128_t ans = 0;
int64_t total_triangles = 0; // Variable to count the number of triangles
mutex mtx;
int r_current = 2;

// Function to convert __int128 to string for output
string int128_to_string(int128_t value) {
    if (value == 0) return "0";
    bool negative = false;
    if (value < 0) {
        negative = true;
        value = -value;
    }
    string result;
    while (value > 0) {
        result = char('0' + value % 10) + result;
        value /= 10;
    }
    if (negative) result = "-" + result;
    return result;
}

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

        int128_t local_ans = 0;
        int64_t local_count = 0; // Local count of triangles

        for (int128_t a = 1; a * a <= 3 * r * r; ++a) {
            for (int128_t b = max((int128_t)(r * r / a), a); a * b <= 3 * r * r; ++b) {
                int128_t p = (int128_t)r * r * (a + b);
                int128_t q = a * b - (int128_t)r * r;
                if (q <= 0)
                    continue;
                if (p < b * q)
                    break;
                if (p % q == 0) {
                    int128_t c = p / q;
                    int128_t perimeter = 2 * (a + b + c);
                    local_ans += perimeter;
                    local_count++; // Increment local triangle count

                    // Uncomment the following line to see the values of a+b, a+c, and b+c
                    // cout << "r = " << r << ": a+b = " << int128_to_string(a + b)
                    //      << ", a+c = " << int128_to_string(a + c)
                    //      << ", b+c = " << int128_to_string(b + c) << endl;
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
    cout << "Total perimeter sum: " << int128_to_string(ans) << endl;
    cout << "Execution time: " << duration << " milliseconds" << endl;

    return 0;
}