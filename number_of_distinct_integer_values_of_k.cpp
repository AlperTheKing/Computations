#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <omp.h>
#include <tuple>

int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

int main() {
    const int MAX = 1000000;
    std::map<int, std::tuple<int, int, int>> k_map; // Map to store distinct k values with one corresponding (m, n, k) tuple
    std::vector<std::map<int, std::tuple<int, int, int>>> thread_k_map(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (int m = 1; m <= MAX; ++m) {
        int thread_id = omp_get_thread_num();
        for (int n = 1; n <= MAX; ++n) {
            if (gcd(m, n) == 1) {  // Only consider reduced fractions m/n
                int numerator = m * m + n * n;
                int denominator = m * n;
                if (numerator % denominator == 0) {
                    int k = numerator / denominator;
                    if (thread_k_map[thread_id].find(k) == thread_k_map[thread_id].end()) {
                        thread_k_map[thread_id][k] = std::make_tuple(m, n, k);
                    }
                }
            }
        }
    }

    // Merge all thread maps into a single map
    for (const auto& thread_map : thread_k_map) {
        for (const auto& pair : thread_map) {
            if (k_map.find(pair.first) == k_map.end()) {
                k_map[pair.first] = pair.second;
            }
        }
    }

    // Print all distinct k values with one corresponding (m, n, k) tuple
    std::cout << "Distinct k values with one corresponding (m, n, k) tuple:" << std::endl;
    for (const auto& [k, tuple] : k_map) {
        auto [m, n, k_val] = tuple;
        std::cout << "k=" << k << ", m=" << m << ", n=" << n << std::endl;
    }

    std::cout << "Number of distinct integer values of k: " << k_map.size() << std::endl;

    return 0;
}
