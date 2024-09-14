// triangle_search_cuda.cu

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel fonksiyonu
__global__ void triangle_kernel(unsigned long long max_perimeter, unsigned long long* d_counter) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_threads = gridDim.x * blockDim.x;

    unsigned long long max_a = max_perimeter / 3;

    for (unsigned long long a = idx + 1; a <= max_a; a += total_threads) {
        unsigned long long max_b = (max_perimeter - a) / 2;
        for (unsigned long long b = a; b <= max_b; ++b) {
            unsigned long long c_min = b;
            unsigned long long c_max = max_perimeter - a - b;

            if (a + b > c_min) {
                if (a + b <= c_max) {
                    c_max = a + b - 1;
                }

                for (unsigned long long c = c_min; c <= c_max; ++c) {
                    unsigned long long numerator = (a + b) * (a + c);
                    unsigned long long denominator = b * c;

                    if (denominator == 0) continue;

                    if (numerator % denominator == 0) {
                        atomicAdd(d_counter, 1ULL);
                    }
                }
            }
        }
    }
}

int main() {
    const unsigned long long max_perimeter = 100000000ULL;
    unsigned long long valid_triangle_count = 0;
    unsigned long long* d_counter;

    // Zaman ölçümünü başlat
    auto start_time = std::chrono::high_resolution_clock::now();

    // GPU belleğinde sayaç için yer ayırma
    cudaMalloc(&d_counter, sizeof(unsigned long long));
    cudaMemset(d_counter, 0, sizeof(unsigned long long));

    // Blok ve iş parçacığı sayısını belirleme
    int threadsPerBlock = 256;
    int blocksPerGrid = 256;

    // CUDA kernel fonksiyonunu çağırma
    triangle_kernel<<<blocksPerGrid, threadsPerBlock>>>(max_perimeter, d_counter);

    // GPU işlemlerinin tamamlanmasını bekleme
    cudaDeviceSynchronize();

    // Sonucu CPU'ya kopyalama
    cudaMemcpy(&valid_triangle_count, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // GPU belleğini serbest bırakma
    cudaFree(d_counter);

    // Zaman ölçümünü bitir
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Toplam geçerli üçgen sayısı: " << valid_triangle_count << std::endl;
    std::cout << "Çalışma süresi: " << elapsed.count() << " saniye" << std::endl;

    return 0;
}