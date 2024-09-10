#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

using namespace std;

const int N = 8;  // Grid size

// Predefined circle coordinates
vector<pair<int, int>> circle_coordinates = {
    {0, 3}, {0, 5}, {0, 7}, {1, 0}, {1, 2}, {2, 1}, {2, 3}, {2, 6},
    {3, 1}, {3, 5}, {4, 3}, {4, 7}, {5, 2}, {5, 4}, {6, 1}, {6, 6},
    {7, 0}, {7, 2}
};

// Directions: up, right, down, left
int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

// Atomic flag to signal when a solution is found
atomic<bool> solution_found(false);

// Mutex for printing
mutex print_mutex;

// Check if the grid coordinates are valid and not visited
bool is_valid(int x, int y, vector<vector<int>>& path) {
    if (x < 0 || y < 0 || x >= N || y >= N) return false;
    return path[x][y] == 0;
}

// DFS function to find a closed path
bool dfs(int x, int y, int start_x, int start_y, vector<vector<int>>& path, int visited) {
    if (solution_found.load()) return false; // Stop if solution is already found
    
    // If all cells are visited and we return to the start, a solution is found
    if (visited == N * N && x == start_x && y == start_y) return true;

    // Try moving in four directions
    for (int i = 0; i < 4; ++i) {
        int new_x = x + directions[i][0];
        int new_y = y + directions[i][1];

        if (is_valid(new_x, new_y, path)) {
            path[new_x][new_y] = 1;  // Visit new cell
            if (dfs(new_x, new_y, start_x, start_y, path, visited + 1)) {
                return true;
            }
            path[new_x][new_y] = 0;  // Backtrack
        }
    }

    return false;
}

// Thread function for parallel DFS
void parallel_dfs(pair<int, int> start_point) {
    int start_x = start_point.first;
    int start_y = start_point.second;

    vector<vector<int>> path(N, vector<int>(N, 0));
    path[start_x][start_y] = 1;

    if (dfs(start_x, start_y, start_x, start_y, path, 1)) {
        if (!solution_found.exchange(true)) {  // Mark that a solution is found
            lock_guard<mutex> guard(print_mutex);
            cout << "Solution found!" << endl;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    cout << path[i][j] << " ";
                }
                cout << endl;
            }
        }
    }
}

int main() {
    // Get the number of hardware threads available
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Default to 4 if hardware_concurrency() is not available

    // Adjust the number of threads if fewer tasks are available
    int num_tasks = circle_coordinates.size();
    num_threads = min(num_threads, num_tasks);

    vector<thread> threads;
    for (int i = 0; i < num_tasks; ++i) {
        if (solution_found.load()) break;  // Stop launching more threads if a solution is found
        threads.emplace_back(parallel_dfs, circle_coordinates[i]);
        if (threads.size() >= num_threads) {
            // Wait for current threads to finish if thread limit is reached
            for (auto& t : threads) {
                t.join();
            }
            threads.clear();
        }
    }

    // Wait for all remaining threads to complete
    for (auto& t : threads) {
        t.join();
    }

    if (!solution_found.load()) {
        cout << "No solution found!" << endl;
    }

    return 0;
}