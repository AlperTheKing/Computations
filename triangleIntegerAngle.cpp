#include <iostream>
#include <cmath>
#include <unordered_set>
#include <tuple>
#include <algorithm>
#include <cstdint>

// Üçgenlerin duplicate olmaması için özel bir hash fonksiyonu tanımlayalım
struct TriangleHash {
    size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& triangle) const {
        auto [a, b, c] = triangle;
        return std::hash<int64_t>()(a) ^ std::hash<int64_t>()(b) ^ std::hash<int64_t>()(c);
    }
};

// GCD fonksiyonu
int64_t gcd(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// int64_t yazdırma fonksiyonu
void print_int64(int64_t n) {
    std::cout << n;
}

int main() {
    const int64_t MAX_PERIMETER = 100000000;

    // Her üçgen türü için setler
    std::unordered_set<std::tuple<int64_t, int64_t, int64_t>, TriangleHash> right_triangles;
    std::unordered_set<std::tuple<int64_t, int64_t, int64_t>, TriangleHash> angle60_triangles;
    std::unordered_set<std::tuple<int64_t, int64_t, int64_t>, TriangleHash> angle120_triangles;

    int max_m = static_cast<int>(sqrt(static_cast<double>(MAX_PERIMETER))) + 1;

    // 90° açılı üçgenler (sağ üçgenler)
    for (int m = 2; m <= max_m; ++m) {
        for (int n = 1; n < m; ++n) {
            if (((m - n) % 2 == 1) && gcd(m, n) == 1) {
                int64_t a0 = static_cast<int64_t>(m) * m - static_cast<int64_t>(n) * n;
                int64_t b0 = 2LL * m * n;
                int64_t c0 = static_cast<int64_t>(m) * m + static_cast<int64_t>(n) * n;

                if (a0 <= 0 || b0 <= 0 || c0 <= 0) continue;

                int64_t P0 = a0 + b0 + c0;

                for (int64_t k = 1; k * P0 <= MAX_PERIMETER; ++k) {
                    int64_t a = k * a0;
                    int64_t b = k * b0;
                    int64_t c = k * c0;

                    int64_t sides[3] = {a, b, c};
                    std::sort(sides, sides + 3);
                    auto triangle = std::make_tuple(sides[0], sides[1], sides[2]);

                    right_triangles.insert(triangle);
                }
            }
        }
    }

    // 60° açılı üçgenler
    for (int m = 2; m <= max_m; ++m) {
        for (int n = 1; n < m; ++n) {
            if (gcd(m, n) != 1) continue;

            int64_t a0 = static_cast<int64_t>(m) * m - static_cast<int64_t>(n) * n;
            int64_t b0 = 2LL * m * n - static_cast<int64_t>(n) * n;
            int64_t c0 = static_cast<int64_t>(m) * m - static_cast<int64_t>(m) * n + static_cast<int64_t>(n) * n;

            if (a0 <= 0 || b0 <= 0 || c0 <= 0) continue;

            int64_t P0 = a0 + b0 + c0;

            for (int64_t k = 1; k * P0 <= MAX_PERIMETER; ++k) {
                int64_t a = k * a0;
                int64_t b = k * b0;
                int64_t c = k * c0;

                int64_t sides[3] = {a, b, c};
                std::sort(sides, sides + 3);
                auto triangle = std::make_tuple(sides[0], sides[1], sides[2]);

                angle60_triangles.insert(triangle);
            }
        }
    }

    // 60° açılı üçgenlere +2 ekle (1,1,1 ve 2,2,2)
    angle60_triangles.insert({1, 1, 1});
    angle60_triangles.insert({2, 2, 2});

    // 120° açılı üçgenler
    for (int m = 2; m <= max_m; ++m) {
        for (int n = 1; n < m; ++n) {
            if (gcd(m, n) != 1) continue;

            int64_t a0 = static_cast<int64_t>(m) * m - static_cast<int64_t>(n) * n;
            int64_t b0 = 2LL * m * n + static_cast<int64_t>(n) * n;
            int64_t c0 = static_cast<int64_t>(m) * m + static_cast<int64_t>(m) * n + static_cast<int64_t>(n) * n;

            if (a0 <= 0 || b0 <= 0 || c0 <= 0) continue;

            int64_t P0 = a0 + b0 + c0;

            for (int64_t k = 1; k * P0 <= MAX_PERIMETER; ++k) {
                int64_t a = k * a0;
                int64_t b = k * b0;
                int64_t c = k * c0;

                int64_t sides[3] = {a, b, c};
                std::sort(sides, sides + 3);
                auto triangle = std::make_tuple(sides[0], sides[1], sides[2]);

                angle120_triangles.insert(triangle);
            }
        }
    }

    // Sonuçları yazdır
    int64_t right_triangle_count = right_triangles.size();
    int64_t angle60_triangle_count = angle60_triangles.size();
    int64_t angle120_triangle_count = angle120_triangles.size();
    int64_t total_triangles = right_triangle_count + angle60_triangle_count + angle120_triangle_count;

    std::cout << "Number of right-angled triangles: ";
    print_int64(right_triangle_count);
    std::cout << std::endl;

    std::cout << "Number of triangles with a 60-degree angle (including equilateral): ";
    print_int64(angle60_triangle_count);
    std::cout << std::endl;

    std::cout << "Number of triangles with a 120-degree angle: ";
    print_int64(angle120_triangle_count);
    std::cout << std::endl;

    std::cout << "Total number of triangles: ";
    print_int64(total_triangles);
    std::cout << std::endl;

    return 0;
}