#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm> // For std::reverse
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace std::chrono;
using namespace boost::multiprecision;

using uint128_t = uint128_t;

// Custom gcd function for uint128_t
uint128_t gcd(uint128_t a, uint128_t b) {
    while (b != 0) {
        uint128_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Function to compute the least common multiple (LCM) of two numbers
uint128_t lcm(uint128_t a, uint128_t b) {
    if (a == 0 || b == 0) return 0;
    return (a / gcd(a, b)) * b;
}

// Function to print uint128_t numbers
void print_uint128(uint128_t n) {
    cout << n.str();
}

int main() {
    // Start measuring time
    auto start = high_resolution_clock::now();

    const uint128_t N = uint128_t("10000000000000000"); // 10^16
    const vector<uint64_t> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                                     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    const int num_primes = primes.size();
    uint128_t result = 0;

    // Calculate the number of integers divisible by at least 4 distinct primes
    // Iterate over combinations of 4 or more primes
    for (int k = 4; k <= num_primes; ++k) {
        vector<int> indices(k);
        // Initialize indices
        for (int i = 0; i < k; ++i) {
            indices[i] = i;
        }
        bool done = false;
        while (!done) {
            // Calculate LCM of selected primes
            uint128_t multiple = 1;
            bool overflow = false;
            for (int idx : indices) {
                uint128_t p = primes[idx];
                uint128_t prev_multiple = multiple;
                multiple = lcm(multiple, p);
                if (multiple > N || multiple < prev_multiple) {
                    overflow = true;
                    break;
                }
            }

            if (!overflow && multiple <= N && multiple > 0) {
                uint128_t count = (N - 1) / multiple;
                int sign = ((k - 4) % 2 == 0) ? 1 : -1;
                result += sign * count;
            }

            // Generate next combination
            done = true;
            for (int i = k - 1; i >= 0; --i) {
                if (indices[i] < num_primes - k + i) {
                    ++indices[i];
                    for (int j = i + 1; j < k; ++j) {
                        indices[j] = indices[j - 1] + 1;
                    }
                    done = false;
                    break;
                }
            }
        }
    }

    cout << "Number of positive integers less than 10^16 divisible by at least four distinct primes less than 100: ";
    print_uint128(result);
    cout << endl;

    // End measuring time
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds." << endl;

    return 0;
}