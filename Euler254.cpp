#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>

// Include Boost Multiprecision
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace boost::multiprecision;

const int MAX_SF = 150;
const int MAX_DIGITS = 30;
const vector<int> DIGITS = {1,2,3,4,5,6,7,8,9};

vector<cpp_int> factorials(10);
mutex mtx;
unordered_map<int, string> sf_n_min_n;

void computeFactorials() {
    factorials[0] = 1;
    for (int i = 1; i <= 9; ++i) {
        factorials[i] = factorials[i - 1] * i;
    }
}

int sumOfDigits(const cpp_int& n) {
    int sum = 0;
    cpp_int temp = n;
    while (temp > 0) {
        sum += static_cast<int>(temp % 10);
        temp /= 10;
    }
    return sum;
}

void generateCombinations(const string& currentDigits, int depth, int maxDepth, const cpp_int& currentF) {
    if (depth > maxDepth) return;

    int sf_n = sumOfDigits(currentF);

    if (sf_n > MAX_SF) return;

    // Sort digits to form minimal n
    string n = currentDigits;
    sort(n.begin(), n.end());

    {
        lock_guard<mutex> lock(mtx);
        if (sf_n_min_n.find(sf_n) == sf_n_min_n.end() ||
            n.length() < sf_n_min_n[sf_n].length() ||
            (n.length() == sf_n_min_n[sf_n].length() && n < sf_n_min_n[sf_n])) {
            sf_n_min_n[sf_n] = n;
        }
    }

    for (int d : DIGITS) {
        string newDigits = currentDigits + char('0' + d);
        cpp_int newF = currentF + factorials[d];
        generateCombinations(newDigits, depth + 1, maxDepth, newF);
    }
}

int main() {
    auto startTime = chrono::high_resolution_clock::now();

    computeFactorials();

    const int NUM_THREADS = thread::hardware_concurrency();
    vector<thread> threads;

    for (int d : DIGITS) {
        threads.emplace_back([d]() {
            string initialDigit(1, '0' + d);
            cpp_int initialF = factorials[d];
            generateCombinations(initialDigit, 1, MAX_DIGITS, initialF);
        });
    }

    for (thread& t : threads) {
        t.join();
    }

    int total_sg = 0;
    for (int i = 1; i <= MAX_SF; ++i) {
        if (sf_n_min_n.find(i) != sf_n_min_n.end()) {
            string n = sf_n_min_n[i];
            int sg = 0;
            for (char c : n) {
                sg += c - '0';
            }
            total_sg += sg;
            cout << "i = " << i << ", g(i) = " << n << ", sg(i) = " << sg << endl;
        }
    }

    cout << "Total sum of sg(i) for i from 1 to " << MAX_SF << " is " << total_sg << endl;

    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = endTime - startTime;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    return 0;
}