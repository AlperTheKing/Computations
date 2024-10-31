#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

// Define the maximum number of digits
const int MAX_DIGITS = 16;

// Offset to handle negative sums (max sum is 9*16=144)
const int OFFSET = 144;

// Memoization table for count2 (alternating sum = 0)
ll memo_count2[MAX_DIGITS + 1][2 * OFFSET + 1][2][2];
bool computed_count2_table[MAX_DIGITS + 1][2 * OFFSET + 1][2][2];

// Digit DP function for count2
ll digitDP_count2(int pos, int sum, bool tight, bool started, const string &num){
    if(pos == MAX_DIGITS){
        return (sum == 0 && started) ? 1 : 0;
    }
    if(computed_count2_table[pos][sum + OFFSET][tight][started]){
        return memo_count2[pos][sum + OFFSET][tight][started];
    }
    computed_count2_table[pos][sum + OFFSET][tight][started] = true;
    ll res = 0;
    int limit = tight ? (num[pos] - '0') : 9;
    for(int d = 0; d <= limit; d++){
        bool new_tight = tight && (d == limit);
        bool new_started = started || (d != 0);
        // Determine the factor based on position
        // Even positions (0-based): factor = -1
        // Odd positions: factor = +1
        int factor = (pos % 2 == 0) ? -1 : 1;
        int new_sum = sum + (new_started ? d * factor : 0);
        if(new_sum + OFFSET < 0 || new_sum + OFFSET > 2 * OFFSET){
            continue; // Prune impossible sums
        }
        res += digitDP_count2(pos + 1, new_sum, new_tight, new_started, num);
    }
    return memo_count2[pos][sum + OFFSET][tight][started] = res;
}

// Memoization table for count3 (alternating sum = 0 and a0 = 0)
ll memo_count3[MAX_DIGITS + 1][2 * OFFSET + 1][2][2];
bool computed_count3_table[MAX_DIGITS + 1][2 * OFFSET + 1][2][2];

// Digit DP function for count3
ll digitDP_count3(int pos, int sum, bool tight, bool started, const string &num){
    if(pos == MAX_DIGITS){
        return (sum == 0 && started) ? 1 : 0;
    }
    if(computed_count3_table[pos][sum + OFFSET][tight][started]){
        return memo_count3[pos][sum + OFFSET][tight][started];
    }
    computed_count3_table[pos][sum + OFFSET][tight][started] = true;
    ll res = 0;
    int limit = tight ? (num[pos] - '0') : 9;
    for(int d = 0; d <= limit; d++){
        // At the last digit (pos = 15), enforce d = 0
        if(pos == MAX_DIGITS - 1 && d != 0){
            continue;
        }
        bool new_tight = tight && (d == limit);
        bool new_started = started || (d != 0);
        // Determine the factor based on position
        int factor = (pos % 2 == 0) ? -1 : 1;
        int new_sum = sum + (new_started ? d * factor : 0);
        if(new_sum + OFFSET < 0 || new_sum + OFFSET > 2 * OFFSET){
            continue; // Prune impossible sums
        }
        res += digitDP_count3(pos + 1, new_sum, new_tight, new_started, num);
    }
    return memo_count3[pos][sum + OFFSET][tight][started] = res;
}

int main(){
    // Define k = 10^16 -1 as a 16-digit number (all 9's)
    string num = "9999999999999999";
    
    // Initialize memoization tables
    memset(computed_count2_table, 0, sizeof(computed_count2_table));
    memset(memo_count2, 0, sizeof(memo_count2));
    memset(computed_count3_table, 0, sizeof(computed_count3_table));
    memset(memo_count3, 0, sizeof(memo_count3));
    
    // Count1: Numbers with a0 = 0
    ll count1 = 1000000000000000LL; // 10^15
    
    // Count2: Numbers with alternating sum = 0
    ll count2 = digitDP_count2(0, 0, true, false, num);
    
    // Count3: Numbers with a0 = 0 and alternating sum = 0
    ll count3 = digitDP_count3(0, 0, true, false, num);
    
    // Calculate Z(k) using Inclusion-Exclusion
    ll Z = count1 + count2 - count3;
    
    cout << "Z(10^16) = " << Z << endl;
    
    return 0;
}