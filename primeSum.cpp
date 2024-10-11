#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <string>
#include <algorithm>
#include <mutex>

using namespace std;

mutex result_mutex;  // Mutex to protect shared data

// Utility function to convert string to __int128
__int128 string_to_int128(const string& str) {
    __int128 result = 0;
    for (char c : str) {
        result = result * 10 + (c - '0');
    }
    return result;
}

// Utility function to convert __int128 to string
string int128_to_string(__int128 value) {
    if (value == 0) return "0";
    string result;
    bool negative = value < 0;
    if (negative) value = -value;
    while (value > 0) {
        result.push_back('0' + value % 10);
        value /= 10;
    }
    if (negative) result.push_back('-');
    reverse(result.begin(), result.end());
    return result;
}

// Function to perform modular exponentiation for __int128
__int128 mod_exp(__int128 base, __int128 exp, __int128 mod) {
    __int128 result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)  // If exp is odd, multiply base with the result
            result = (result * base) % mod;
        exp = exp >> 1;     // exp = exp / 2
        base = (base * base) % mod;  // base = base^2 % mod
    }
    return result;
}

// Optimized Miller-Rabin primality test for larger numbers
bool miller_rabin(__int128 n, __int128 a) {
    if (n % a == 0) return false;
    __int128 d = n - 1;
    while (d % 2 == 0) d /= 2;
    __int128 x = mod_exp(a, d, n);  // Use mod_exp for modular exponentiation
    if (x == 1 || x == n - 1) return true;
    while (d != n - 1) {
        x = (x * x) % n;
        d *= 2;
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

bool is_prime(__int128 n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    static const vector<__int128> bases = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    for (__int128 a : bases) {
        if (n == a) return true;
        if (!miller_rabin(n, a)) return false;
    }
    return true;
}

// Function to check if N can be represented as a sum of K primes
bool can_be_sum_of_primes(__int128 N, __int128 K) {
    if (K == 1) {
        return is_prime(N);
    }
    if (K == 2) {
        if (N < 4) return false;
        if (N % 2 == 0) return true; // Goldbach's conjecture
        return is_prime(N - 2);
    }
    if (K >= 3) {
        return N >= 2 * K;
    }
    return false;
}

void process_test_cases(const vector<pair<__int128, __int128>>& test_cases, vector<string>& results, atomic<size_t>& index) {
    size_t i;
    while ((i = index++) < test_cases.size()) {
        __int128 N = test_cases[i].first;
        __int128 K = test_cases[i].second;
        string result = can_be_sum_of_primes(N, K) ? "Yes" : "No";
        
        // Mutex lock for thread-safe access to results
        lock_guard<mutex> lock(result_mutex);
        results[i] = result;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    size_t T;
    cin >> T;
    vector<pair<__int128, __int128>> test_cases(T);
    for (size_t i = 0; i < T; ++i) {
        string n_str, k_str;
        cin >> n_str >> k_str;
        test_cases[i].first = string_to_int128(n_str);
        test_cases[i].second = string_to_int128(k_str);
    }

    vector<string> results(T);
    atomic<size_t> index(0);

    // Determine the number of threads to use
    unsigned int num_threads = thread::hardware_concurrency();
    vector<thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(process_test_cases, cref(test_cases), ref(results), ref(index));
    }
    for (auto& t : threads) {
        t.join();
    }

    for (const auto& result : results) {
        cout << result << "\n";
    }

    return 0;
}