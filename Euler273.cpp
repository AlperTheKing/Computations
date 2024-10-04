#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <set>
#include <map>
#include <boost/multiprecision/cpp_int.hpp>
#include <sstream>

namespace mp = boost::multiprecision;

std::mutex mtx;        // Mutex for global sum
std::mutex file_mtx;   // Mutex for file writing

using BigInt = mp::cpp_int;

// Multiply two Gaussian integers (a + ib) * (c + id)
std::pair<BigInt, BigInt> multiplyGaussian(const BigInt& a1, const BigInt& b1, const BigInt& a2, const BigInt& b2) {
    BigInt real = a1 * a2 - b1 * b2;
    BigInt imag = a1 * b2 + a2 * b1;
    return {real, imag};
}

// Generate all combinations of Gaussian integer representations
void computeRepresentations(const std::vector<std::pair<int, int>>& primeReps, int index,
                            std::pair<BigInt, BigInt> current,
                            std::set<std::pair<BigInt, BigInt>>& results) {
    if (index == primeReps.size()) {
        BigInt a = mp::abs(current.first);
        BigInt b = mp::abs(current.second);
        if (a > b) std::swap(a, b);
        results.insert({a, b});
        return;
    }
    // Include the prime representation in positive form
    auto [a1, b1] = primeReps[index];
    auto [r1, i1] = multiplyGaussian(current.first, current.second, a1, b1);
    computeRepresentations(primeReps, index + 1, {r1, i1}, results);

    // Include the prime representation in negative form
    auto [r2, i2] = multiplyGaussian(current.first, current.second, a1, -b1);
    computeRepresentations(primeReps, index + 1, {r2, i2}, results);
}

// Thread function to process a subset of Ns
void processNs(const std::vector<std::pair<BigInt, std::vector<int>>>& Ns_with_factors,
               const std::map<int, std::pair<int, int>>& primeReps,
               mp::cpp_int& sumOfAs,
               std::map<BigInt, std::vector<std::pair<BigInt, BigInt>>>& localResults) {
    mp::cpp_int localSumOfAs = 0;

    for (const auto& [N, factors] : Ns_with_factors) {
        // Get representations for each prime factor
        std::vector<std::pair<int, int>> reps;
        for (int p : factors) {
            reps.push_back(primeReps.at(p));
        }

        // Compute all combinations of representations
        std::set<std::pair<BigInt, BigInt>> representations;
        computeRepresentations(reps, 0, {1, 0}, representations);

        // Add representations to local results
        localResults[N] = std::vector<std::pair<BigInt, BigInt>>(representations.begin(), representations.end());

        // Sum a-values
        for (const auto& [a, b] : representations) {
            localSumOfAs += a;
        }
    }

    // Lock and update global sum
    {
        std::lock_guard<std::mutex> lock(mtx);
        sumOfAs += localSumOfAs;
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Precomputed list of primes of the form 4k + 1 less than 150 and their (a, b) representations with a <= b
    std::vector<int> primes_4k1 = {
        5, 13, 17, 29, 37, 41, 53, 61,
        73, 89, 97, 101, 109, 113, 137, 149
    };

    std::map<int, std::pair<int, int>> primeReps = {
        {5,   {1, 2}},
        {13,  {2, 3}},
        {17,  {1, 4}},
        {29,  {2, 5}},
        {37,  {1, 6}},
        {41,  {4, 5}},
        {53,  {2, 7}},
        {61,  {5, 6}},
        {73,  {3, 8}},
        {89,  {5, 8}},
        {97,  {4, 9}},
        {101, {1, 10}},
        {109, {3, 10}},
        {113, {7, 8}},
        {137, {4, 11}},
        {149, {7, 10}}
    };

    // Step 3: Generate all square-free Ns (products of distinct primes)
    std::vector<std::pair<BigInt, std::vector<int>>> Ns_with_factors;
    int totalPrimes = primes_4k1.size();
    mp::cpp_int numCombinations = mp::pow(mp::cpp_int(2), totalPrimes); // 2^n combinations

    for (mp::cpp_int i = 1; i < numCombinations; ++i) {
        BigInt N = 1;
        std::vector<int> factors;
        for (int j = 0; j < totalPrimes; ++j) {
            if (i & (mp::cpp_int(1) << j)) {
                N *= primes_4k1[j];
                factors.push_back(primes_4k1[j]);
            }
        }
        Ns_with_factors.push_back({N, factors});
    }

    // Sort Ns in ascending order
    std::sort(Ns_with_factors.begin(), Ns_with_factors.end(),
              [](const std::pair<BigInt, std::vector<int>>& a, const std::pair<BigInt, std::vector<int>>& b) {
                  return a.first < b.first;
              });

    // Step 4: Divide Ns among threads
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Default to 4 threads if unable to detect
    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<BigInt, std::vector<int>>>> Ns_per_thread(numThreads);

    for (size_t i = 0; i < Ns_with_factors.size(); ++i) {
        Ns_per_thread[i % numThreads].push_back(Ns_with_factors[i]);
    }

    mp::cpp_int sumOfAs = 0;
    std::vector<std::map<BigInt, std::vector<std::pair<BigInt, BigInt>>>> threadResults(numThreads);

    // Start threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back(processNs, std::cref(Ns_per_thread[t]), std::cref(primeReps), std::ref(sumOfAs), std::ref(threadResults[t]));
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Combine results from all threads
    std::map<BigInt, std::vector<std::pair<BigInt, BigInt>>> combinedResults;
    for (const auto& localResult : threadResults) {
        for (const auto& [N, reps] : localResult) {
            combinedResults[N].insert(combinedResults[N].end(), reps.begin(), reps.end());
        }
    }

    // Sort N values in ascending order
    std::vector<BigInt> sortedNs;
    for (const auto& [N, _] : combinedResults) {
        sortedNs.push_back(N);
    }
    std::sort(sortedNs.begin(), sortedNs.end());

    // Write combined results to file
    {
        std::ofstream outfile("SumofSquares.txt");
        for (const auto& N : sortedNs) {
            outfile << N << ", ";
            const auto& reps = combinedResults[N];
            for (size_t i = 0; i < reps.size(); ++i) {
                outfile << "(" << reps[i].first << "," << reps[i].second << ")";
                if (i != reps.size() - 1) {
                    outfile << ", ";
                }
            }
            outfile << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the sum of a-values and execution time
    std::cout << "Sum of a-values: " << sumOfAs << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}