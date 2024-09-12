#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    const long long max_sum = 100000000; // a + b + c <= 100 milyon
    int count = 0;

    // Zaman ölçümünü başlat
    double start_time = omp_get_wtime();

    // OpenMP ile paralel döngü
    #pragma omp parallel for reduction(+:count) schedule(dynamic)
    for (int a = 1; a < max_sum / 3; ++a) {  // a kenarı
        for (int b = a; b < (max_sum - a) / 2; ++b) {  // b kenarı
            int c = max_sum - a - b;  // c kenarı
            if (c < b) continue;

            // Üçgen olma şartını kontrol et
            if (a + b > c && a + c > b && b + c > a) {
                // (a + b + c) / (b * c) tam sayı mı?
                if ((a + b + c) % (b * c) == 0) {
                    #pragma omp critical
                    {
                        cout << "Valid triangle: a = " << a << ", b = " << b << ", c = " << c << endl;
                    }
                    count++; // Geçerli bir üçgen bulduk
                }
            }
        }
    }

    // Zaman ölçümünü bitir
    double end_time = omp_get_wtime();

    // Toplam üçgen sayısını ve çalışma süresini yazdır
    cout << "Total number of valid triangles: " << count << endl;
    cout << "Execution time: " << (end_time - start_time) << " seconds" << endl;

    return 0;
}