#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define MAX_RESULTS 100000000 // Maksimum üçgen sayısı (gerekirse artırılabilir)

struct Triangle {
    long long a, b, c;
};

// GPU'ya taşımak için Triangle yapısını düz bir dizi olarak kullanacağız
struct TriangleGPU {
    long long a;
    long long b;
    long long c;
};

// Newton yöntemiyle tam sayı karekökü hesaplayan cihaz fonksiyonu
__device__ long long integerSqrt(long long n) {
    if (n == 0 || n == 1)
        return n;
    long long x = n;
    long long y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// GPU kernel fonksiyonu
__global__ void findTrianglesKernel(int k, long long MAX_PERIMETER, TriangleGPU* d_results, unsigned long long* d_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalThreads = gridDim.x * blockDim.x;

    long long m = 2 + 4 * (k - 1); // m = 6 for k=2, m=10 for k=3

    for (long long b = idx + 1; b <= MAX_PERIMETER / 2; b += totalThreads) {
        for (long long c = b; c <= MAX_PERIMETER - b - 1; ++c) {
            // Compute discriminant D
            __int128 D = (__int128)b * b + (__int128)c * c + (__int128)m * b * c;
            long long s = integerSqrt(D);
            if ((long long)s * s != D) continue; // Not a perfect square
            // Compute 'a'
            long long a_numerator = - (b + c) + s;
            if (a_numerator <= 0 || a_numerator % 2 != 0) continue;
            long long a = a_numerator / 2;
            if (a > b) continue; // Ensure a <= b
            // Check triangle inequalities
            if (a + b <= c || a + c <= b || b + c <= a) continue;
            // Check perimeter constraint
            if (a + b + c > MAX_PERIMETER) continue;
            // Verify that the ratio equals 'k'
            long long ratio_numerator = (a + b) * (a + c);
            long long ratio_denominator = b * c;
            if (ratio_numerator != k * ratio_denominator) continue;
            // Atomically store the valid triangle
            unsigned long long pos = atomicAdd(d_count, 1ULL);
            if (pos < MAX_RESULTS) {
                d_results[pos].a = a;
                d_results[pos].b = b;
                d_results[pos].c = c;
            }
        }
    }
}

// Equilateral triangles için ayrı bir kernel fonksiyonu
__global__ void findEquilateralTrianglesKernel(long long MAX_PERIMETER, TriangleGPU* d_results, unsigned long long* d_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalThreads = gridDim.x * blockDim.x;

    for (long long a = idx + 1; 3 * a <= MAX_PERIMETER; a += totalThreads) {
        // Atomically store the valid triangle
        unsigned long long pos = atomicAdd(d_count, 1ULL);
        if (pos < MAX_RESULTS) {
            d_results[pos].a = a;
            d_results[pos].b = a;
            d_results[pos].c = a;
        }
    }
}

int main() {
    // Kullanıcıdan maksimum çevre değerini al
    long long MAX_PERIMETER;
    std::cout << "Enter the maximum perimeter: ";
    std::cin >> MAX_PERIMETER;

    auto start_time = std::chrono::high_resolution_clock::now();

    // CUDA ayarları
    int device = 0;
    cudaSetDevice(device);

    int threadsPerBlock = 256;
    int blocksPerGrid = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    blocksPerGrid = prop.multiProcessorCount * 32; // Her SM'de 32 blok

    // Sonuçları depolamak için bellek tahsisi
    TriangleGPU* d_results;
    cudaMalloc((void**)&d_results, MAX_RESULTS * sizeof(TriangleGPU));

    unsigned long long* d_count;
    cudaMalloc((void**)&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    // k = 2 ve k = 3 için kernel'ları başlat
    findTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(2, MAX_PERIMETER, d_results, d_count);
    findTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(3, MAX_PERIMETER, d_results, d_count);

    // k = 4 (equilateral triangles) için kernel'ı başlat
    findEquilateralTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(MAX_PERIMETER, d_results, d_count);

    // Kernel işlemlerinin tamamlanmasını bekle
    cudaDeviceSynchronize();

    // Sonuç sayısını al
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (h_count > MAX_RESULTS) h_count = MAX_RESULTS;

    // Sonuçları host tarafına kopyala
    std::vector<Triangle> validTriangles(h_count);
    TriangleGPU* h_results = new TriangleGPU[h_count];
    cudaMemcpy(h_results, d_results, h_count * sizeof(TriangleGPU), cudaMemcpyDeviceToHost);

    // TriangleGPU dizisini Triangle yapısına dönüştür
    for (unsigned long long i = 0; i < h_count; ++i) {
        validTriangles[i].a = h_results[i].a;
        validTriangles[i].b = h_results[i].b;
        validTriangles[i].c = h_results[i].c;
    }

    // Bellek temizliği
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_count);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Sonuçları göster
    std::cout << "Found " << h_count << " valid triangles.\n";
    std::cout << "Time taken: " << elapsed.count() << " seconds.\n";

    // İsteğe bağlı olarak üçgenleri yazdırabilirsiniz
    /*
    for (const auto& triangle : validTriangles) {
        std::cout << "a = " << triangle.a << ", b = " << triangle.b << ", c = " << triangle.c << "\n";
    }
    */

    return 0;
}