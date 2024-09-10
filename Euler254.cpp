#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>

using namespace std;

// Precompute factorials of digits 0-9
vector<int> factorials = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};

// Function to calculate the sum of factorials of digits of n (f(n))
int f(int n) {
    int sum = 0;
    while (n > 0) {
        sum += factorials[n % 10]; // Add factorial of the last digit
        n /= 10;
    }
    return sum;
}

// Function to calculate the sum of digits of a number (sf(n))
int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Function to find the smallest n such that sf(f(n)) = i
int g(int i, int &f_val, int &sf_val) {
    int n = 1;
    while (true) {
        f_val = f(n);        // Calculate f(n)
        sf_val = sum_of_digits(f_val);  // Calculate sf(f(n))
        if (sf_val == i) {
            return n;
        }
        n++;
    }
}

// Function to calculate the sum of digits of g(i)
int sg(int gi) {
    return sum_of_digits(gi);
}

int main() {
    int upper_limit = 20;
    int total_sum = 0;

    // Loop through all i from 1 to upper_limit (i.e., 1 to 20)
    for (int i = 1; i <= upper_limit; i++) {
        int f_val = 0, sf_val = 0;  // To store f(n) and sf(f(n)) values
        int gi = g(i, f_val, sf_val);  // Get g(i), along with f(n) and sf(f(n))
        int sgi = sg(gi);  // Get sg(i)

        // Output all the required values
        cout << "i = " << i << ":\n";
        cout << "  g(" << i << ") = " << gi << "\n";
        cout << "  f(g(" << i << ")) = " << f_val << "\n";
        cout << "  sf(f(g(" << i << "))) = " << sf_val << "\n";
        cout << "  sg(" << i << ") = " << sgi << "\n";

        // Add sg(i) to the total sum
        total_sum += sgi;
    }

    // Output the total sum of sg(i) for 1 <= i <= upper_limit
    cout << "Sum of sg(i) for 1 <= i <= " << upper_limit << " is: " << total_sum << endl;

    return 0;
}