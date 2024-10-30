#include <iostream>
#include <set>
#include <vector>
#include <cmath> // for std::sqrt
#include <numeric> // for std::gcd
#include <omp.h>

const int MAX_PERIMETER = 100000000;
const int max_m = 50000; // Set max_m to 50,000 as per user preference

struct Triangle {
    int a, b, c;
    bool operator<(const Triangle& other) const {
        return std::tie(a, b, c) < std::tie(other.a, other.b, other.c);
    }
    int perimeter() const {
        return a + b + c;
    }
};

// Global sets to store unique primitive triangles for each type
std::set<Triangle> primitives60;
std::set<Triangle> primitives90;
std::set<Triangle> primitives120;

void addPrimitiveTriangle(std::set<Triangle>& localPrimitives, int a, int b, int c) {
    int gcd = std::gcd(std::gcd(a, b), c);
    a /= gcd;
    b /= gcd;
    c /= gcd;
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    localPrimitives.insert({a, b, c});
}

// Generator functions for 60°, 90°, and 120° triangles
void triangleGen(int type, int& total, std::set<Triangle>& globalPrimitives) {
    std::set<Triangle> localPrimitives;
    int localTotal = 0;

    #pragma omp parallel for schedule(dynamic, 100) reduction(+:localTotal) // Dynamic scheduling with chunk size of 100
    for (int m = 2; m <= max_m; ++m) {
        for (int n = 1; n < m; ++n) {
            int a, b, c;

            // Generate triangles based on the type
            if (type == 60) {
                a = m * m - n * n;
                b = 2 * m * n - n * n;
                c = m * m - m * n + n * n;
            } else if (type == 90) {
                a = m * m - n * n;
                b = 2 * m * n;
                c = m * m + n * n;
            } else if (type == 120) {
                a = m * m - n * n;
                b = 2 * m * n + n * n;
                c = m * m + m * n + n * n;
            }

            if (a + b + c <= MAX_PERIMETER) {
                #pragma omp critical
                {
                    addPrimitiveTriangle(localPrimitives, a, b, c);
                }
            }
        }
    }

    // Calculate the number of scaled triangles for each local primitive
    for (const auto& triangle : localPrimitives) {
        int perimeter = triangle.perimeter();
        int scaledCount = MAX_PERIMETER / perimeter;
        localTotal += scaledCount;
    }

    // Update the global set and total with a critical section to avoid race conditions
    #pragma omp critical
    {
        globalPrimitives.insert(localPrimitives.begin(), localPrimitives.end());
        total += localTotal;
    }
}

int main() {
    int total60 = 0;
    int total90 = 0;
    int total120 = 0;

    auto start = omp_get_wtime();

    // Generate triangles in parallel for each type
    #pragma omp parallel sections
    {
        #pragma omp section
        triangleGen(60, total60, primitives60);

        #pragma omp section
        triangleGen(90, total90, primitives90);

        #pragma omp section
        triangleGen(120, total120, primitives120);
    }

    int totalTriangles = total60 + total90 + total120;

    auto end = omp_get_wtime();

    // Display the results
    std::cout << "\nTotal number of triangles: " << totalTriangles << std::endl;
    std::cout << "60° triangles: " << total60 << std::endl;
    std::cout << "90° triangles: " << total90 << std::endl;
    std::cout << "120° triangles: " << total120 << std::endl;
    std::cout << "Time taken: " << (end - start) << " seconds" << std::endl;

    return 0;
}