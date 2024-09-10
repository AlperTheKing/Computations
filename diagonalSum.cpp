#include <iostream>
#include <vector>
#include <algorithm>
#include <climits> // INT_MAX

const int N = 5;
int grid[N][N]; // 5x5 grid
int min_sum_global = INT_MAX;
int diagonal_positions[5][2] = {{0, 4}, {1, 3}, {2, 2}, {3, 1}, {4, 0}};
int best_grid[N][N] = {0}; // En iyi grid çözümünü saklamak için

// Yönler: Yukarı, Aşağı, Sol, Sağ
int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

// Grid'in durumu yazdırma
void print_grid(int grid[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << grid[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Diyagonal toplamını hesaplama
int calc_diagonal_sum(int grid[N][N]) {
    int sum = 0;
    for (int i = 0; i < 5; ++i) {
        int row = diagonal_positions[i][0];
        int col = diagonal_positions[i][1];
        sum += grid[row][col];
    }
    return sum;
}

// Sayı grid'de nereye yerleştirilebilir diye kontrol eden fonksiyon
bool is_safe(int row, int col, int number) {
    // Eğer grid dolu değilse
    if (grid[row][col] != 0)
        return false;

    // Eğer 1 ise ilk sayıyı herhangi bir yere yerleştirebiliriz
    if (number == 1)
        return true;

    // Bir önceki sayının komşu karede olup olmadığını kontrol et
    for (int i = 0; i < 4; ++i) {
        int new_row = row + directions[i][0];
        int new_col = col + directions[i][1];

        if (new_row >= 0 && new_row < N && new_col >= 0 && new_col < N && grid[new_row][new_col] == number - 1) {
            return true;
        }
    }
    return false;
}

// Backtracking ile grid'e sayıları yerleştiren fonksiyon
void solve(int number) {
    // 25'e kadar olan sayıları yerleştirdik mi?
    if (number == 26) {
        int current_sum = calc_diagonal_sum(grid);
        if (current_sum < min_sum_global) {
            min_sum_global = current_sum;
            // En iyi çözümü sakla
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    best_grid[i][j] = grid[i][j];
                }
            }
        }
        return;
    }

    // Grid üzerinde tüm hücreleri dene
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (is_safe(i, j, number)) {
                // Sayıyı yerleştir
                grid[i][j] = number;

                // Bir sonraki sayıyı yerleştir
                solve(number + 1);

                // Backtrack (geri adım at), bu hücreyi boşalt
                grid[i][j] = 0;
            }
        }
    }
}

int main() {
    // Grid'i sıfırla
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            grid[i][j] = 0;

    // Backtracking ile çözümü başlat
    solve(1);

    // Sonuçları yazdır
    std::cout << "Minimum diagonal sum: " << min_sum_global << std::endl;
    std::cout << "Best grid configuration:" << std::endl;
    print_grid(best_grid);

    return 0;
}