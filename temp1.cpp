#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <omp.h>  // OpenMP için gerekli kütüphane

// Faktöriyel hesaplama fonksiyonu
int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; ++i) {
        fact *= i;
    }
    return fact;
}

// Bir sayının basamaklarının faktöriyel toplamını hesaplayan fonksiyon (f(n))
int f(int n, const std::vector<int>& factorials) {
    int sum = 0;
    while (n > 0) {
        sum += factorials[n % 10]; // Basamak faktöriyelini diziden al
        n /= 10;
    }
    return sum;
}

// Bir sayının basamaklarının toplamını hesaplayan fonksiyon (sf(n))
int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Verileri parça parça işleyip dosyaya yazan fonksiyon
void process_chunk(long long start, long long end, const std::vector<int>& factorials, std::unordered_map<int, std::pair<long long, int>>& sf_map) {
    #pragma omp parallel for
    for (long long n = start; n <= end; ++n) {
        int fn = f(n, factorials);          // f(n) hesapla
        int sfn = sum_of_digits(fn);  // sf(n) hesapla

        // Map'e ekleme işlemini kritik bölgeye alıyoruz
        #pragma omp critical
        {
            // Eğer sf(n) değeri daha önce eklenmemişse, ekle
            // Eğer sf(n) değeri varsa, daha küçük n değerine sahip olanı tut
            if (sf_map.find(sfn) == sf_map.end() || n < sf_map[sfn].first) {
                sf_map[sfn] = {n, fn};  // sf(n) -> (n, f(n)) şeklinde kaydet
            }
        }
    }
}

int main() {
    // 0'dan 9'a kadar olan sayıların faktöriyel değerlerini hesaplayalım
    std::vector<int> factorials(10);
    
    // Faktöriyel hesaplamasını yapalım
    for (int i = 0; i <= 9; ++i) {
        factorials[i] = factorial(i);
    }

    // Verileri parça parça işlemek için dilim büyüklüğünü tanımlayalım
    long long chunk_size = 1000000000; // 1 milyar (10^9) büyüklüğünde parçalar
    long long total = 1000000000000;   // 1 trilyon (10^12)

    // Her parça için map kullanarak en küçük n'yi tutacağız
    std::unordered_map<int, std::pair<long long, int>> sf_map;

    // Dilimleri işleyip dosyaya yazalım
    for (long long start = 0; start < total; start += chunk_size) {
        long long end = std::min(start + chunk_size - 1, total - 1);
        process_chunk(start, end, factorials, sf_map);

        // Her dilimden sonra verileri dosyaya yazalım
        std::ofstream outfile("sf(n)_sorted_unique.txt", std::ios::app);
        if (!outfile) {
            std::cerr << "Dosya açılamadı!" << std::endl;
            return 1;
        }

        // sf(n) değerlerine göre sıralamak için verileri bir vektöre aktaralım
        std::vector<std::tuple<int, long long, int>> sorted_values;
        for (const auto& entry : sf_map) {
            int sfn = entry.first;
            long long n = entry.second.first;
            int fn = entry.second.second;
            sorted_values.push_back(std::make_tuple(sfn, n, fn));
        }

        // sf(n) değerlerine göre sıralama yapalım
        std::sort(sorted_values.begin(), sorted_values.end());

        // Sıralı ve tekrar etmeyen sf(n) değerlerini dosyaya yazalım
        for (const auto& entry : sorted_values) {
            int sfn;
            long long n;
            int fn;
            std::tie(sfn, n, fn) = entry;  // Tuple'dan verileri al
            outfile << "n: " << n << ", f(n): " << fn << ", sf(n): " << sfn << std::endl;
        }

        // Dosya kapat
        outfile.close();

        // Geçici map'i temizleyelim
        sf_map.clear();
    }

    std::cout << "Sıralı ve tekrarsız sf(n) değerleri en küçük n ile sf(n)_sorted_unique.txt dosyasına başarıyla yazdırıldı." << std::endl;

    return 0;
}