#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

using namespace std;

struct Point {
    int r, c;
};

int N_TARGET = 10; 
atomic<bool> solved(false);
mutex print_mutex;
vector<int> solution;

bool are_collinear(const Point& p1, const Point& p2, const Point& p3) {
    long long val = (long long)(p2.c - p1.c) * (p3.r - p2.r) - 
                    (long long)(p2.r - p1.r) * (p3.c - p2.c);
    return val == 0;
}

bool is_safe(const vector<Point>& placed, int r, int c) {
    for (const auto& p : placed) {
        if (p.c == c) return false;
        if (abs(p.r - r) == abs(p.c - c)) return false;
    }

    int n_placed = placed.size();
    for (int i = 0; i < n_placed; ++i) {
        for (int j = i + 1; j < n_placed; ++j) {
            if (are_collinear(placed[i], placed[j], {r, c})) {
                return false;
            }
        }
    }
    return true;
}

void solve_worker(int n, int seed) {
    mt19937 rng(seed);
    
    while (!solved) {
        vector<Point> placement;
        placement.reserve(n);
        
        vector<int> cols(n);
        iota(cols.begin(), cols.end(), 1);
        
        vector<int> row_order(n);
        iota(row_order.begin(), row_order.end(), 0);
        
        bool restart = false;

        for (int r : row_order) {
            if (solved) return;
            
            shuffle(cols.begin(), cols.end(), rng);
            
            bool placed_row = false;
            for (int c : cols) {
                if (is_safe(placement, r, c)) {
                    placement.push_back({r, c});
                    placed_row = true;
                    break;
                }
            }
            
            if (!placed_row) {
                restart = true;
                break;
            }
        }

        if (!restart && placement.size() == n) {
            lock_guard<mutex> lock(print_mutex);
            if (!solved) {
                solved = true;
                solution.resize(n);
                for(const auto& p : placement) {
                    solution[p.r] = p.c;
                }
            }
            return;
        }
    }
}

int main() {
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    vector<thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(solve_worker, N_TARGET, i + std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    if (solved) {
        cout << N_TARGET << endl;
        for (int i = 0; i < N_TARGET; ++i) {
            cout << solution[i] << (i == N_TARGET - 1 ? "" : " ");
        }
        cout << endl;
    } else {
        cout << "No solution found." << endl;
    }
    
    return 0;
}