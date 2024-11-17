#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
const int MOD = 1000000007;

// Precompute powers of 10 modulo MOD
ll pow10_mod[20];
void precompute_powers() {
    pow10_mod[0] = 1;
    for(int i=1;i<20;i++) {
        pow10_mod[i] = (pow10_mod[i-1] * 10) % MOD;
    }
}

// Helper function to convert number to string with leading zeros to make it 'len' digits
string to_padded_string(ll num, int len) {
    string s = to_string(num);
    while(s.size() < len) s = "0" + s;
    return s;
}

// DP memoization tables
ll dp_count[20][2][2][163][163];
ll dp_sum[20][2][2][163][163];
bool computed_dp[20][2][2][163][163];

// Digit DP function for step 2: sum of numbers of exact length 'len' where sum_first_k == sum_last_k
pair<ll, ll> digitDP(int pos, bool tight_low, bool tight_high, bool leading_zero, int sum_first_k, int sum_last_k, int len, int k, const string &lb, const string &ub) {
    // Base case: all digits processed
    if(pos == len) {
        if(leading_zero) {
            // No number formed
            return {0, 0};
        }
        if(sum_first_k == sum_last_k) {
            // Valid k-balanced number
            return {1, 0}; // Sum is accumulated in recursion
        }
        else {
            return {0, 0};
        }
    }

    // Memoization check
    if(computed_dp[pos][tight_low][tight_high][sum_first_k][sum_last_k]) {
        return {dp_count[pos][tight_low][tight_high][sum_first_k][sum_last_k],
                dp_sum[pos][tight_low][tight_high][sum_first_k][sum_last_k]};
    }

    // Mark as computed
    computed_dp[pos][tight_low][tight_high][sum_first_k][sum_last_k] = true;

    ll cnt = 0, sm = 0;

    // Determine the range of digits to iterate
    int lower = tight_low ? (lb[pos] - '0') : 0;
    int upper = tight_high ? (ub[pos] - '0') : 9;

    for(int digit = lower; digit <= upper; digit++) {
        // Update tight constraints
        bool new_tight_low = tight_low && (digit == lower);
        bool new_tight_high = tight_high && (digit == upper);
        bool new_leading_zero = leading_zero && (digit == 0);

        // Update sum_first_k and sum_last_k
        int new_sum_first_k = sum_first_k;
        int new_sum_last_k = sum_last_k;

        if(!new_leading_zero) {
            if(pos < k) {
                new_sum_first_k += digit;
            }
            if(pos >= len - k) {
                new_sum_last_k += digit;
            }
        }

        // Recursive call
        pair<ll, ll> temp = digitDP(pos + 1, new_tight_low, new_tight_high, new_leading_zero, new_sum_first_k, new_sum_last_k, len, k, lb, ub);

        // Calculate the positional contribution of the current digit
        ll digit_contribution = (digit * pow10_mod[len - pos - 1]) % MOD;

        // Accumulate the sum
        sm = (sm + (digit_contribution * temp.first) % MOD + temp.second) % MOD;
        cnt = (cnt + temp.first) % MOD;
    }

    // Store in memoization tables
    dp_count[pos][tight_low][tight_high][sum_first_k][sum_last_k] = cnt;
    dp_sum[pos][tight_low][tight_high][sum_first_k][sum_last_k] = sm;

    return {cnt, sm};
}

// Function to compute the sum of k-balanced numbers of exact length 'len' within [L, R]
ll sum_k_balanced_exact_length(ll L_num, ll R_num, int len, int k) {
    // Compute lower and upper bounds for numbers of length 'len' within [L, R]
    ll lower_l = max(L_num, (len == 1) ? 0LL : (ll)pow(10, len -1));
    ll upper_l = min(R_num, (ll)pow(10, len) -1);

    if(lower_l > upper_l) return 0;

    // Convert lower_l and upper_l to 'len'-digit strings with leading zeros
    string lb = to_padded_string(lower_l, len);
    string ub = to_padded_string(upper_l, len);

    // Reset memoization tables
    memset(computed_dp, 0, sizeof(computed_dp));

    // Perform Digit DP
    pair<ll, ll> res = digitDP(0, true, true, true, 0, 0, len, k, lb, ub);

    return res.second; // Return the sum
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);

    precompute_powers();

    ll L_num, R_num;
    int k;
    cin >> L_num >> R_num >> k;

    // Step 1: Sum of numbers with length <=k
    ll sum_step1 = 0;
    for(int len=1; len<=k; len++) {
        // Compute lower and upper bounds for numbers of length 'len'
        ll lower_l = max(L_num, (len ==1) ? 0LL : (ll)pow(10, len -1));
        ll upper_l = min(R_num, (ll)pow(10, len) -1);

        if(lower_l > upper_l) continue;

        // Sum of numbers from lower_l to upper_l using arithmetic progression formula
        ll count = upper_l - lower_l +1;
        ll total_sum = ((lower_l + upper_l) % MOD) * (count % MOD) % MOD;
        // To divide by 2 under modulo, multiply by inverse of 2 modulo MOD
        ll inv2 = 500000004; // Since MOD is prime, inv2 = 2^{MOD-2} mod MOD
        total_sum = (total_sum * inv2) % MOD;
        sum_step1 = (sum_step1 + total_sum) % MOD;
    }

    // Step 2: Sum of numbers with length >k and sum_first_k == sum_last_k
    ll sum_step2 = 0;
    for(int len=k+1; len<=18; len++) {
        // Compute lower and upper bounds for numbers of length 'len'
        ll lower_l = max(L_num, (ll)(len ==1 ? 0 : pow(10, len -1)));
        ll upper_l = min(R_num, (ll)(len ==18 ? LLONG_MAX : pow(10, len) -1));

        if(lower_l > upper_l) continue;

        // Compute the sum using Digit DP
        ll current_sum = sum_k_balanced_exact_length(L_num, R_num, len, k);
        sum_step2 = (sum_step2 + current_sum) % MOD;
    }

    // Final sum
    ll S = (sum_step1 + sum_step2) % MOD;

    cout << S << "\n";
    return 0;
}