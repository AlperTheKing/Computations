#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// GPU'da bitwise method ile karekök bulan fonksiyon
__device__ long long bitwise_sqrt(long long n) {
    if (n == 0 || n == 1) return n;
    
    long long x = n;
    long long result = 0;
    long long bit = 1LL << 62;  // Başlangıçta en büyük bit'i ayarla (bitwise shift ile)
    
    while (bit > x) bit >>= 2;  // İlk uygun bit'i bul
    
    // Karekökü bitwise hesapla
    while (bit != 0) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    
    return result;
}

// GPU'da çalışacak kernel fonksiyonu
__global__ void find_solutions(long long* Ns, long long* a_sums, int numNs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numNs) return; // Eğer index N sayısından büyükse geri dön

    long long N = Ns[idx]; // Her thread kendi N değerini alır
    long long sum_a = 0;

    // a ve b değerlerini bulma
    for (int a = 0; a * a <= N; ++a) {
        long long b_squared = N - static_cast<long long>(a) * a;
        long long b = bitwise_sqrt(b_squared);  // Bitwise method ile karekök bul
        if (b * b == b_squared && a <= b) {
            sum_a += a;
        }
    }

    // Toplam a değerini global hafızaya yaz
    a_sums[idx] = sum_a;
}

// CPU üzerinde prime üretim fonksiyonu
std::vector<int> generate_primes(int limit) {
    std::vector<int> primes;
    for (int i = 5; i <= limit; i += 4) {
        bool is_prime = true;
        for (int j = 2; j <= std::sqrt(i); j++) {
            if (i % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            primes.push_back(i);
        }
    }
    return primes;
}

// CPU üzerinde kombinasyonlarla N üretme fonksiyonu
void generate_square_free_N(const std::vector<int>& primes, std::vector<long long>& square_free_N) {
    size_t num_primes = primes.size();
    for (size_t i = 1; i < (1 << num_primes); ++i) {
        long long N = 1;
        for (size_t j = 0; j < num_primes; ++j) {
            if (i & (1 << j)) {
                N *= primes[j];
            }
        }
        square_free_N.push_back(N);
    }
}

// Toplam a değerini bulmak için CUDA kodu
int main() {
    // Zamanı ölçmek için başlangıç
    auto start_time = std::chrono::high_resolution_clock::now();

    // Prime limit
    int prime_limit = 150;

    // Prime'ları üret
    std::vector<int> primes = generate_primes(prime_limit);

    // Kare-siz N değerlerini üret
    std::vector<long long> square_free_N;
    generate_square_free_N(primes, square_free_N);

    size_t num_Ns = square_free_N.size();

    // CUDA'ya kopyalanacak diziler
    long long* d_Ns;
    long long* d_a_sums;
    
    // CUDA'daki blok ve thread yapılandırmasını bulmak için cudaOccupancyMaxPotentialBlockSize kullan
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, find_solutions, 0, num_Ns);
    int gridSize = (num_Ns + blockSize - 1) / blockSize;

    // CUDA için alan ayırma
    cudaMalloc(&d_Ns, num_Ns * sizeof(long long));
    cudaMalloc(&d_a_sums, num_Ns * sizeof(long long));

    // N dizisini CUDA'ya kopyala
    cudaMemcpy(d_Ns, square_free_N.data(), num_Ns * sizeof(long long), cudaMemcpyHostToDevice);

    // Çözüm kernel'ini başlat
    find_solutions<<<gridSize, blockSize>>>(d_Ns, d_a_sums, num_Ns);

    // Sonuçları ana belleğe geri kopyala
    std::vector<long long> a_sums(num_Ns);
    cudaMemcpy(a_sums.data(), d_a_sums, num_Ns * sizeof(long long), cudaMemcpyDeviceToHost);

    // Toplam a değerini hesapla
    long long total_a_sum = 0;
    for (long long sum : a_sums) {
        total_a_sum += sum;
    }

    // CUDA'da kullanılan alanı temizle
    cudaFree(d_Ns);
    cudaFree(d_a_sums);

    // Zamanı ölçmek için bitiş
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Toplam a değerini ve geçen süreyi ekrana yazdır
    std::cout << "Total sum of a values: " << total_a_sum << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}