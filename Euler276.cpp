#include <bits/stdc++.h>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <atomic>

using namespace std;

// Function to convert __int128 to string
string int128_to_str(__int128 n) {
    bool is_negative = false;
    if (n < 0) {
        is_negative = true;
        n = -n;
    }

    string s = "";
    if (n == 0) {
        s = "0";
    } else {
        while (n > 0) {
            char digit = '0' + (n % 10);
            s += digit;
            n /= 10;
        }
        if (is_negative) {
            s += '-';
        }
        reverse(s.begin(), s.end());
    }
    return s;
}

int main() {
    // Start time measurement
    auto start_time = chrono::steady_clock::now();

    // Define the perimeter limit
    const int N = 10000000; // 10,000,000

    // Step 1: Compute Alcuin's Sequence a(n) for n =3 to N via Dynamic Programming
    // a(n) = number of integer-sided triangles with perimeter n
    // This is equivalent to the number of solutions to 2k +3m +4p = n -3
    // where k, m, p >=0

    // Initialize dp array with unsigned long long to prevent overflow
    // dp[s] = number of solutions to 2k +3m +4p = s
    // s ranges from 0 to N-3
    vector<unsigned long long> dp(N - 2, 0); // s=0 to N-3, size N-2+1=N-2
    dp[0] = 1; // Base case: one way to have sum=0

    // Define the coins
    vector<int> coins = {2, 3, 4};

    for(auto coin : coins){
        for(int s = coin; s <= N -3; s++){
            dp[s] += dp[s - coin];
        }
    }

    // Now, a(n) = dp[n -3] for n >=3
    // Initialize prefix sum array S(n) = sum_{k=3}^n a(k)
    // Using __int128 to store large sums
    vector<__int128> S(N +1, 0); // S[0] to S[N]

    for(int n=3; n<=N; n++){
        S[n] = S[n-1] + dp[n -3];
    }

    // Step 2: Compute Möbius function μ(d) for d =1 to N via Sieve of Eratosthenes
    vector<int> mu(N +1, 1); // μ(0) is unused
    vector<bool> is_prime_mu(N +1, true);
    is_prime_mu[0] = is_prime_mu[1] = false;

    for(int p=2; p<=N; p++){
        if(is_prime_mu[p]){
            // p is prime
            for(int multiple = p; multiple <=N; multiple +=p){
                mu[multiple] *= -1;
                is_prime_mu[multiple] = false;
            }
            // Set μ(d) =0 for multiples of p^2
            long long p_sq = static_cast<long long>(p) * p;
            if(p_sq > N) continue;
            for(long long multiple = p_sq; multiple <=N; multiple +=p_sq){
                mu[multiple] =0;
            }
        }
    }

    // Step 3: Compute the total number of primitive triangles using Möbius inversion
    // Total = sum_{d=1}^{N} μ(d) * S(floor(N /d))
    // Implemented using multithreading with dynamic load balancing

    // Determine the number of available threads
    unsigned int num_threads = thread::hardware_concurrency();
    if(num_threads ==0) num_threads =4; // Fallback to 4 threads if unable to detect

    // Define the chunk size for dynamic load balancing
    const int chunk_size = 100000; // Adjust based on performance/testing

    // Shared variables
    __int128 total_sum =0;
    mutex sum_mutex;

    // Atomic counter for dynamic load balancing
    atomic<int> current_d(1);

    // Lambda function for worker threads
    auto worker = [&](int thread_id){
        __int128 partial_sum =0;
        while(true){
            int start_d = current_d.fetch_add(chunk_size);
            if(start_d >N) break;
            int end_d = min(start_d + chunk_size -1, N);
            for(int d=start_d; d<=end_d; d++){
                int k = N /d;
                if(k <3){
                    // S[k] for k <3 is 0, since a(n) =0 for n <3
                    continue;
                }
                partial_sum += static_cast<__int128>(mu[d]) * S[k];
            }
        }
        // Safely add the partial_sum to the total_sum
        lock_guard<mutex> guard(sum_mutex);
        total_sum += partial_sum;
    };

    // Launch threads
    vector<thread> threads;
    for(unsigned int t=0; t<num_threads; t++){
        threads.emplace_back(worker, t);
    }

    // Join threads
    for(auto &th: threads){
        th.join();
    }

    // End time measurement
    auto end_time = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Step 4: Output the result
    cout << "Total number of primitive integer-sided triangles with perimeter <= " << N << " is:\n";
    cout << int128_to_str(total_sum) << "\n";

    // Step 5: Output the elapsed time
    cout << "Computation Time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}