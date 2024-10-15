#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <numeric>  // For gcd function
using namespace std;

namespace PE261 {
    const uint64_t N = 1e14;

    // Function to calculate the integer square root
    uint64_t integerSqrt(uint64_t x) {
        return static_cast<uint64_t>(sqrt(x));
    }

    // Function to generate values based on u and d
    vector<uint64_t> generate(uint64_t u, uint64_t d) {
        vector<uint64_t> ret;
        uint64_t v = integerSqrt((u * u - 1) / d);
        for (uint64_t x = u, y = v; x <= N; ) {
            uint64_t p = x * u + d * y * v;
            uint64_t q = x * v + y * u;
            ret.push_back((x - 1) >> 1);
            x = p;
            y = q;
        }
        return ret;
    }

    // Function to compute the core value by removing square factors
    uint64_t core(uint64_t v) {
        for (uint64_t i = 2; i * i <= v; ++i) {
            while (v % (i * i) == 0) {
                v /= i * i;  // Remove square factors
            }
        }
        return v;
    }

    // Function to calculate a custom square root adjusted by the GCD
    uint64_t Sqrt(uint64_t a, uint64_t d) {
        uint64_t gcd_a_d = gcd(a, d);  // Use std::gcd from <numeric>
        uint64_t gcd_a1_d = gcd(a + 1, d);
        return integerSqrt(a / gcd_a_d) * integerSqrt((a + 1) / gcd_a1_d);
    }

    // Main function to find square-pivots and calculate their sum
    void main() {
        uint64_t count = 0;
        set<uint64_t> unique_k_values;  // Set to track unique k values

        // Iterate over potential values of u
        for (uint64_t u = 1; u * u <= N * 0.55; ++u) {
            uint64_t d = core(u) * core(u + 1);  // Calculate core value for u and u+1

            auto vec = generate(u * 2 + 1, d);  // Generate potential k values

            // Iterate over all generated pairs of values (a, b)
            for (auto a : vec) {
                for (auto b : vec) {
                    if (b > a) break;  // Ensure b <= a for valid combinations

                    uint64_t s = Sqrt(a, d) * Sqrt(b, d) * d;
                    uint64_t k = s + (a + 1) * b;  // Calculate k
                    uint64_t m = b;                // m is set to b
                    uint64_t n = s + a * (b + 1);  // Calculate n

                    if (k > N) break;  // Stop if k exceeds the limit

                    unique_k_values.insert(k);  // Insert k into the set

                    // Print u, v, k, m, n values for debugging/verification
                    // cout << "Found: u = " << a << ", v = " << b 
                    //      << ", k = " << k << ", m = " << m << ", n = " << n << endl;
                }
            }
        }

        // Output the total count of unique k values
        cout << "Total count of unique k values = " << unique_k_values.size() << endl;

        // Calculate the sum of all unique k values
        uint64_t total_sum = 0;
        for (const auto& k : unique_k_values) {
            total_sum += k;
        }

        // Output the final result
        cout << "Sum of all unique k values = " << total_sum << endl;
    }
}

int main() {
    PE261::main();
    return 0;
}