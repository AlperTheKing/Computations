#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

// Function to count the power of a prime `p` in the factorial of `n`
int power_of_p_in_factorial(long long n, int p) {
    int count = 0;
    while (n > 0) {
        n /= p;
        count += n;
    }
    return count;
}

vector<int> binomial_coefficient_powers(long long N, int P) {
    int power_in_N_fact = power_of_p_in_factorial(N, P);
    unordered_map<int, int> counts;
    int max_L = 0;

    int power_in_K_fact = 0;
    int power_in_N_minus_K_fact;

    for (long long K = 0; K <= N; ++K) {
        if (K > 0) {
            // Calculate power of P in K! incrementally
            power_in_K_fact += power_of_p_in_factorial(K, P);
        }
        // Power of P in (N-K)! by reusing power in N! and subtracting power in K!
        power_in_N_minus_K_fact = power_in_N_fact - power_in_K_fact;

        // Calculate power of P in C(N, K)
        int power = power_in_N_fact - (power_in_K_fact + power_in_N_minus_K_fact);
        
        if (power >= 0) {
            counts[power]++;
            max_L = max(max_L, power);
        }
    }

    vector<int> result(max_L + 1, 0);
    for (int i = 0; i <= max_L; ++i) {
        result[i] = counts[i];
    }
    return result;
}

void solve() {
    int T;
    cin >> T;

    for (int i = 0; i < T; ++i) {
        long long N;
        int P;
        cin >> N >> P;
        vector<int> result = binomial_coefficient_powers(N, P);

        for (size_t j = 0; j < result.size(); ++j) {
            if (j > 0) cout << " ";
            cout << result[j];
        }
        cout << endl;
    }
}

int main() {
    ios::sync_with_stdio(false); // Fast I/O
    cin.tie(nullptr);

    solve();
    return 0;
}