#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Fonksiyon, çarpma işlemini kontrol eder
bool isValid(int A, int B, int C, int D) {
    int AABC = 1100 * A + 10 * B + C;
    int AA = 11 * A;
    int result = AABC * AA;
    
    // DDDDDD'nin 6 basamaklı olup olmadığını kontrol et
    if (result / 100000 == D && result / 10000 % 10 == D && result / 1000 % 10 == D &&
        result / 100 % 10 == D && result / 10 % 10 == D && result % 10 == D) {
        return true;
    }
    return false;
}

int main() {
    // Zaman ölçümünü başlat
    auto start = high_resolution_clock::now();
    
    // Harflerin alabileceği rakamlar 0-9 arasındadır
    for (int A = 1; A <= 9; A++) {  // A sıfır olamaz çünkü başta
        for (int B = 0; B <= 9; B++) {
            if (B == A) continue;
            for (int C = 0; C <= 9; C++) {
                if (C == A || C == B) continue;
                for (int D = 1; D <= 9; D++) { // D sıfır olamaz çünkü DDDDDD altı basamaklı
                    if (D == A || D == B || D == C) continue;
                    
                    if (isValid(A, B, C, D)) {
                        cout << "Solution found: A = " << A << ", B = " << B << ", C = " << C << ", D = " << D << endl;
                    }
                }
            }
        }
    }
    
    // Zaman ölçümünü bitir
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;
    return 0;
}