#include <iostream>
#include <vector>
#include <string>
using namespace std;

typedef long long ll;

// Function to perform matrix multiplication modulo M
void matmul(ll a[2][2], ll b[2][2], ll M) {
    ll res[2][2];
    res[0][0] = (a[0][0] * b[0][0] + a[0][1] * b[1][0]) % M;
    res[0][1] = (a[0][0] * b[0][1] + a[0][1] * b[1][1]) % M;
    res[1][0] = (a[1][0] * b[0][0] + a[1][1] * b[1][0]) % M;
    res[1][1] = (a[1][0] * b[0][1] + a[1][1] * b[1][1]) % M;

    // Copy result back to matrix a
    a[0][0] = res[0][0];
    a[0][1] = res[0][1];
    a[1][0] = res[1][0];
    a[1][1] = res[1][1];
}

// Function to compute matrix exponentiation modulo M
void mat_pow(ll mat[2][2], ll n, ll M) {
    ll result[2][2] = { {1, 0}, {0, 1} }; // Identity matrix

    while (n > 0) {
        if (n & 1) {
            matmul(result, mat, M);
        }
        matmul(mat, mat, M);
        n >>= 1;
    }

    // Copy result back to mat
    mat[0][0] = result[0][0];
    mat[0][1] = result[0][1];
    mat[1][0] = result[1][0];
    mat[1][1] = result[1][1];
}

// Function to compute X mod M using matrix exponentiation
ll compute_X_mod_M(ll A, ll N, ll M) {
    if (N == 0 || M == 1) return 0;

    // Calculate the number of digits in A
    int L = to_string(A).length();

    // Calculate r = 10^L % M
    ll r = 1 % M;
    ll base = 10 % M;
    while (L > 0) {
        if (L & 1) {
            r = (r * base) % M;
        }
        base = (base * base) % M;
        L >>= 1;
    }

    // Initialize the transformation matrix
    ll mat[2][2] = { {r % M, A % M}, {0, 1} };

    // Raise the matrix to the N-th power
    mat_pow(mat, N, M);

    // The result is in mat[0][1]
    ll X_mod_M = mat[0][1] % M;
    return X_mod_M;
}

int main() {
    int T;
    cin >> T;

    vector<ll> A_list(T), N_list(T), M_list(T);

    // Read all inputs first
    for (int i = 0; i < T; ++i) {
        cin >> A_list[i] >> N_list[i] >> M_list[i];
    }

    // Compute and output results
    for (int i = 0; i < T; ++i) {
        ll A = A_list[i];
        ll N = N_list[i];
        ll M = M_list[i];

        ll result = compute_X_mod_M(A, N, M);
        cout << result << endl;
    }

    return 0;
}