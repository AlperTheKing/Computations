#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>

const int LIMIT = 1000;
std::mutex mtx;
std::vector<std::tuple<int, int, int, int>> solutions;

void findSolutions(int start, int end) {
    for (int x = start; x <= end; ++x) {
        for (int y = x; y <= LIMIT; ++y) {
            for (int z = y; z <= LIMIT; ++z) {
                int a = std::cbrt(x * x * x + y * y * y + z * z * z);
                if (a <= LIMIT && a * a * a == x * x * x + y * y * y + z * z * z) {
                    std::lock_guard<std::mutex> lock(mtx);
                    solutions.emplace_back(x, y, z, a);
                }
            }
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = LIMIT / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size + 1;
        int end = (i == num_threads - 1) ? LIMIT : (i + 1) * chunk_size;
        threads.emplace_back(findSolutions, start, end);
    }

    for (auto &t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    for (const auto &[x, y, z, a] : solutions) {
        std::cout << "x: " << x << ", y: " << y << ", z: " << z << ", a: " << a << '\n';
    }
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}