#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>

struct Triplet {
    int a, b, c;
};

int main() {
    int count = 0;
    std::vector<Triplet> triplets;

    // Zaman ölçümünü başlat
    auto start = std::chrono::high_resolution_clock::now();

    // OpenMP parallelization
    #pragma omp parallel for collapse(3) reduction(+:count)
    for (int a = 0; a <= 110000000; ++a) {
        for (int b = 0; b <= 110000000; ++b) {
            for (int c = 0; c <= 110000000; ++c) {
                if (8*a*a*a + 15*a*a + 6*a - 27*b*b*c == 1) {
                    double expr1 = std::cbrt(a + b * std::sqrt(c));
                    double expr2 = std::cbrt(a - b * std::sqrt(c));
                    if (expr1 + expr2 == 1) {
                        #pragma omp critical
                        {
                            triplets.push_back({a, b, c});
                        }
                        count++;
                    }
                }
            }
        }
    }

    // Tripletleri 'a' değerine göre sırala
    std::sort(triplets.begin(), triplets.end(), [](const Triplet& t1, const Triplet& t2) {
        return t1.a < t2.a;
    });

    // Zaman ölçümünü bitir
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Sıralanmış tripletleri yazdır
    for (const auto& triplet : triplets) {
        std::cout << triplet.a << " " << triplet.b << " " << triplet.c << std::endl;
    }

    std::cout << count << " triplet(s) found" << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}