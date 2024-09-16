#include <iostream>
#include <vector>
#include <chrono>

const int MOD = 20092010;
const int SIZE = 2000;  // Matrisin boyutu

using Matrix = std::vector<std::vector<long long>>;

// Matris çarpma fonksiyonu (tek çekirdek)
Matrix mat_mult(const Matrix& A, const Matrix& B) {
    Matrix C(SIZE, std::vector<long long>(SIZE, 0));

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                C[i][j] = (C[i][j] + (A[i][k] * B[k][j]) % MOD) % MOD;
            }
        }
    }

    return C;
}

// Matris üs alma fonksiyonu (üs alma yöntemi)
Matrix mat_pow(Matrix base, long long exp) {
    Matrix result(SIZE, std::vector<long long>(SIZE, 0));
    for (int i = 0; i < SIZE; ++i) result[i][i] = 1;  // Birim matris
    
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = mat_mult(result, base);
        }
        base = mat_mult(base, base);
        exp /= 2;
    }
    return result;
}

// Matris-vektör çarpma fonksiyonu
std::vector<long long> mat_vec_mult(const Matrix& A, const std::vector<long long>& v) {
    std::vector<long long> result(SIZE, 0);
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            result[i] = (result[i] + A[i][j] * v[j]) % MOD;
        }
    }
    return result;
}

// Çözüm fonksiyonu
long long solve(long long N) {
    // Farklı bir çözüm yöntemi olan matrix üs alma ile çözme
    std::vector<long long> g(SIZE, 1);  // İlk 2000 değer 1

    Matrix base(SIZE, std::vector<long long>(SIZE, 0));

    // Temel matris oluşturma
    for (int i = 0; i < SIZE - 1; ++i) {
        base[i][i + 1] = 1;
    }
    base[SIZE - 1][0] = 1;
    base[SIZE - 1][1] = 1;

    if (N <= SIZE) {
        return g[N - 1];
    }

    // N - SIZE adımı için matris üs alma
    Matrix result = mat_pow(base, N - SIZE);

    // İlk 2000 değeri g vektörü ile çarpma
    std::vector<long long> final_vec = mat_vec_mult(result, g);

    return final_vec[SIZE - 1];
}

int main() {
    long long N = 1000000000000000000;  // N = 10^18

    // Zaman ölçümünü başlat
    auto start = std::chrono::high_resolution_clock::now();

    long long result = solve(N);

    // Zaman ölçümünü bitir
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Sonuç ve geçen süreyi yazdır
    std::cout << "Final result for N = " << N << ": " << result << std::endl;
    std::cout << "Total time taken: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}