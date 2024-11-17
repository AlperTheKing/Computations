#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

// Constants
const long long MOD = 1000000000;

// Function to compute divisor sums using Sieve of Eratosthenes with multithreading
void compute_divisor_sums(long long N, vector<long long> &divisor_sums, long long start, long long end) {
    for(long long i = start; i <= end; ++i){
        for(long long j = i; j <= N; j += i){
            divisor_sums[j] += i;
        }
    }
}

int main() {
    // Prompt user for input
    long long N;
    cout << "Enter the value of N: ";
    cin >> N;

    // Start time measurement
    auto start_time = chrono::high_resolution_clock::now();

    // Initialize divisor sums up to N^2
    // To manage memory, we'll compute d(k) on the fly for each k*j
    // Instead of storing d(k) for all k up to N^2, we'll compute d(k) using prime factorization
    // However, for simplicity and efficiency, we'll precompute d(k) up to N^2 for small N

    // Check if N is too large
    if(N > 10000){
        cout << "N is too large for this implementation. Please enter N <= 10000." << endl;
        return 0;
    }

    long long max_kj = N * N;
    // Initialize divisor sums
    vector<long long> d(max_kj + 1, 0);

    // Multithreading parameters
    unsigned int num_threads = thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4; // Default to 4 if unable to detect
    vector<thread> threads;
    long long block_size = max_kj / num_threads;
    if(block_size == 0) block_size = 1;

    // Launch threads to compute divisor sums
    for(unsigned int t = 0; t < num_threads; ++t){
        long long start_idx = t * block_size + 1;
        long long end_idx = (t == num_threads -1) ? max_kj : (t + 1) * block_size;
        threads.emplace_back(compute_divisor_sums, max_kj, ref(d), start_idx, end_idx);
    }

    // Join threads
    for(auto &th : threads){
        th.join();
    }

    // Clear thread vector for reuse
    threads.clear();

    // Compute S(N) with multithreading
    // Split the work among threads by rows (i from 1 to N)
    long long S = 0;
    mutex mtx;
    long long rows_per_thread = N / num_threads;
    if(rows_per_thread == 0) rows_per_thread = 1;

    auto compute_S = [&](long long start_i, long long end_i) {
        long long local_sum = 0;
        for(long long i = start_i; i <= end_i; ++i){
            for(long long j = 1; j <= N; ++j){
                long long kj = i * j;
                if(kj <= max_kj){
                    local_sum += d[kj];
                    if(local_sum >= MOD){
                        local_sum %= MOD;
                    }
                }
            }
        }
        // Lock and update the global sum
        lock_guard<mutex> lock(mtx);
        S = (S + local_sum) % MOD;
    };

    // Launch threads to compute S(N)
    for(unsigned int t = 0; t < num_threads; ++t){
        long long start_i = t * rows_per_thread + 1;
        long long end_i = (t == num_threads -1) ? N : (t + 1) * rows_per_thread;
        threads.emplace_back(compute_S, start_i, end_i);
    }

    // Join threads
    for(auto &th : threads){
        th.join();
    }

    // End time measurement
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    // Output the result
    cout << "S(" << N << ") mod 10^9 = " << S << endl;
    cout << "Time taken: " << elapsed.count() << " seconds." << endl;

    return 0;
}