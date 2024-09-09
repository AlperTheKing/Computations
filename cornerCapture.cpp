#include <iostream>
#include <vector>
#include <pthread.h>
#include <cstdlib>

using namespace std;

const int N = 8;  // Grid boyutu

// Verilen çember koordinatları
vector<pair<int, int>> circle_coordinates = {
    {0, 3}, {0, 5}, {0, 7}, {1, 0}, {1, 2}, {2, 1}, {2, 3}, {2, 6},
    {3, 1}, {3, 5}, {4, 3}, {4, 7}, {5, 2}, {5, 4}, {6, 1}, {6, 6},
    {7, 0}, {7, 2}
};

// Dört yön: yukarı, sağ, aşağı, sol
int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

// Grid kontrolü
bool is_valid(int x, int y, vector<vector<int>>& path) {
    // Hücre grid sınırları içinde mi ve zaten ziyaret edilmemiş mi?
    if (x < 0 || y < 0 || x >= N || y >= N) return false;
    return path[x][y] == 0;
}

// DFS fonksiyonu, kapalı bir yol bulmaya çalışır
bool dfs(int x, int y, int start_x, int start_y, vector<vector<int>>& path, int visited) {
    // Eğer tüm hücreler gezildiyse ve başlama noktasına döndüysek çözüm bulundu demektir
    if (visited == N * N && x == start_x && y == start_y) return true;

    // Dört yön boyunca ilerlemeye çalış
    for (int i = 0; i < 4; ++i) {
        int new_x = x + directions[i][0];
        int new_y = y + directions[i][1];

        if (is_valid(new_x, new_y, path)) {
            path[new_x][new_y] = 1;  // Yeni hücreyi ziyaret et
            if (dfs(new_x, new_y, start_x, start_y, path, visited + 1)) {
                return true;
            }
            path[new_x][new_y] = 0;  // Geri dön (backtrack)
        }
    }

    return false;
}

// DFS işlemini başlatmak için yapı
struct ThreadArgs {
    pair<int, int> start_point;
};

// Thread fonksiyonu
void* parallel_dfs(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*)args;
    int start_x = threadArgs->start_point.first;
    int start_y = threadArgs->start_point.second;

    vector<vector<int>> path(N, vector<int>(N, 0));
    path[start_x][start_y] = 1;

    if (dfs(start_x, start_y, start_x, start_y, path, 1)) {
        cout << "Çözüm bulundu!" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << path[i][j] << " ";
            }
            cout << endl;
        }
        exit(0);  // İlk çözüm bulunduğunda programı bitir
    }

    pthread_exit(NULL);
}

int main() {
    int num_threads = circle_coordinates.size();
    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    // Her çember koordinatından ayrı bir thread başlatılır
    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].start_point = circle_coordinates[i];
        pthread_create(&threads[i], NULL, parallel_dfs, (void*)&thread_args[i]);
    }

    // Tüm threadlerin tamamlanmasını bekleyelim
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    cout << "Çözüm bulunamadı!" << endl;
    return 0;
}