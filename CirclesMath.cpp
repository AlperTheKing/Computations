#include <bits/stdc++.h>
using namespace std;

// Constants
const int MOD = 1000000007;
const int MAX = 1000000;

// Precomputed arrays
long long factorial_arr[MAX + 1];
long long inv_factorial_arr[MAX + 1];
long long derangements_arr[MAX + 1];

// Function to compute x^y mod MOD using binary exponentiation
long long power_mod(long long x, long long y, long long mod_val) {
    long long res = 1;
    x %= mod_val;
    while (y > 0) {
        if (y & 1LL) {
            res = res * x % mod_val;
        }
        x = x * x % mod_val;
        y >>= 1LL;
    }
    return res;
}

// Function to precompute factorials and inverse factorials
void precompute_factorials() {
    factorial_arr[0] = 1;
    for(int i = 1; i <= MAX; ++i){
        factorial_arr[i] = factorial_arr[i-1] * i % MOD;
    }
    // Compute inverse factorial using Fermat's Little Theorem
    inv_factorial_arr[MAX] = power_mod(factorial_arr[MAX], MOD-2, MOD);
    for(int i = MAX-1; i >=0; --i){
        inv_factorial_arr[i] = inv_factorial_arr[i+1] * (i+1) % MOD;
    }
}

// Function to precompute derangements
void precompute_derangements() {
    derangements_arr[0] = 1;
    if(MAX >=1){
        derangements_arr[1] = 0;
    }
    for(int i=2;i<=MAX;i++){
        derangements_arr[i] = ((i-1) * (derangements_arr[i-1] + derangements_arr[i-2])) % MOD;
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    // Precompute factorials, inverse factorials, and derangements
    precompute_factorials();
    precompute_derangements();
    
    // Read number of test cases
    int T;
    cin >> T;
    
    // Vector to store all test cases
    vector<pair<int, int>> test_cases(T);
    for(int i=0;i<T;i++) {
        cin >> test_cases[i].first >> test_cases[i].second; // n and k
    }
    
    // Process each test case
    for(int i=0;i<T;i++) {
        int n = test_cases[i].first;
        int k = test_cases[i].second;
        
        // Handle special cases
        if(k > n || k < 2){
            cout << "0\n";
            continue;
        }
        
        long long total = 0;
        // Iterate over m from 1 to floor(n/k)
        int max_m = n / k;
        for(int m=1; m<=max_m; m++) {
            // Compute the term: (-1)^{m+1} * (n! / ((n - m*k)! * m! * k^m)) * D(n - m*k)
            // Calculate n! * inv[(n - m*k)!] % MOD
            long long term = factorial_arr[n];
            term = term * inv_factorial_arr[n - m*k] % MOD;
            // Multiply by inv_factorial[m]
            term = term * inv_factorial_arr[m] % MOD;
            // Compute k^m mod MOD
            long long k_pow_m = power_mod(k, m, MOD);
            // Compute inverse of k^m mod MOD
            long long inv_k_pow_m = power_mod(k_pow_m, MOD-2, MOD);
            term = term * inv_k_pow_m % MOD;
            // Multiply by derangements[n - m*k]
            term = term * derangements_arr[n - m*k] % MOD;
            // Apply inclusion-exclusion
            if(m % 2 == 1){
                total = (total + term) % MOD;
            }
            else{
                total = (total - term + MOD) % MOD;
            }
        }
        cout << total << "\n";
    }
    
    return 0;
}