#include <bits/stdc++.h>
using namespace std;

// Function to check if a number is a perfect square
bool isPerfectSquare(long long num) {
    if(num < 0) return false;
    long long root = sqrt((double)num);
    return root * root == num;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    long long n;
    cin >> n;
    
    // Precompute non-perfect squares up to n
    vector<int> nonPerfectSquares;
    nonPerfectSquares.reserve(n);
    for(int i=1; i<=n; i++){
        int root = sqrt(i);
        if(root * root != i){
            nonPerfectSquares.push_back(i);
        }
    }
    
    long long F_n = 0;
    
    // Iterate over y and z
    for(auto y : nonPerfectSquares){
        for(auto z : nonPerfectSquares){
            long long s_sq = (long long)y * y - z;
            if(s_sq <= 0) continue;
            if(!isPerfectSquare(s_sq)) continue;
            long long s = sqrt(s_sq);
            // Check if (y + s) and (y - s) are both even
            if( (y + s) % 2 != 0 || (y - s) % 2 != 0 ) continue;
            long long p = (y + s) / 2;
            long long q = (y - s) / 2;
            if(p < 0 || q < 0) continue;
            // Now, x + p + q must be a perfect square.
            // Thus, x = k^2 - p - q, where 1 <= x <=n
            // So, k must satisfy:
            // k^2 - p - q >=1  => k >= ceil(sqrt(p + q +1))
            // k^2 - p - q <=n  => k <= floor(sqrt(p + q +n))
            double tmp_min = sqrt( (double)(p + q + 1) );
            long long k_min = ceil(tmp_min);
            double tmp_max = sqrt( (double)(p + q + n) );
            long long k_max = floor(tmp_max);
            if(k_min > k_max) continue;
            // The number of valid k's is (k_max - k_min +1)
            F_n += (k_max - k_min +1);
        }
    }
    
    cout << F_n << "\n";
    
    return 0;
}