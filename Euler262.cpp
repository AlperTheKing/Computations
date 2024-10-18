#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include <thread>
#include <mutex>

using namespace std;
using namespace std::chrono;

// Grid parameters
const double GRID_STEP = 0.05;  // Adjusted grid resolution
const int MAX_COORD = 1600;
const int GRID_SIZE = MAX_COORD / GRID_STEP + 1;

// Starting and ending grid indices
const int AX = static_cast<int>(200 / GRID_STEP);
const int AY = static_cast<int>(200 / GRID_STEP);
const int BX = static_cast<int>(1400 / GRID_STEP);
const int BY = static_cast<int>(1400 / GRID_STEP);

// Movement directions and costs
const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
const double move_cost[8] = {GRID_STEP, GRID_STEP, GRID_STEP, GRID_STEP,
                             GRID_STEP * sqrt(2), GRID_STEP * sqrt(2),
                             GRID_STEP * sqrt(2), GRID_STEP * sqrt(2)};

// Mutex for thread safety
mutex mtx;

// Function to compute h(x, y)
double compute_h(int xi, int yi) {
    double x = xi * GRID_STEP;
    double y = yi * GRID_STEP;

    double numerator = x * x + y * y + x * y;
    double part1 = 5000.0 - numerator / 200.0 + 25.0 * pow(x + y, 2.0);
    double exponent = -((x * x + y * y) / 1000000.0 - 3.0 * (x + y) / 2000.0 + 0.7);
    double h = part1 * exp(exponent);
    return h;
}

// Heuristic function for A* algorithm
double heuristic(int xi, int yi) {
    double x = xi * GRID_STEP;
    double y = yi * GRID_STEP;
    double bx = BX * GRID_STEP;
    double by = BY * GRID_STEP;
    return sqrt((x - bx) * (x - bx) + (y - by) * (y - by));
}

// Function to compute h(x, y) in parallel
void compute_h_parallel(vector<vector<double>>& h, int start_row, int end_row, double& local_max_h, double& local_min_h, mutex& mtx) {
    double local_max = -numeric_limits<double>::infinity();
    double local_min = numeric_limits<double>::infinity();

    for (int xi = start_row; xi < end_row; ++xi) {
        for (int yi = 0; yi < GRID_SIZE; ++yi) {
            h[xi][yi] = compute_h(xi, yi);
            if (h[xi][yi] > local_max) local_max = h[xi][yi];
            if (h[xi][yi] < local_min) local_min = h[xi][yi];
        }
    }

    // Update global max and min h values
    lock_guard<mutex> lock(mtx);
    if (local_max > local_max_h) local_max_h = local_max;
    if (local_min < local_min_h) local_min_h = local_min;
}

int main() {
    auto start_time = high_resolution_clock::now();

    // Step 1: Compute h(x, y) for all grid points in parallel
    vector<vector<double>> h(GRID_SIZE, vector<double>(GRID_SIZE));
    double max_h = -numeric_limits<double>::infinity();
    double min_h = numeric_limits<double>::infinity();

    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Default to 4 threads if hardware_concurrency() returns 0
    vector<thread> threads;
    int rows_per_thread = GRID_SIZE / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? GRID_SIZE : start_row + rows_per_thread;
        threads.emplace_back(compute_h_parallel, ref(h), start_row, end_row, ref(max_h), ref(min_h), ref(mtx));
    }

    for (auto& th : threads) {
        th.join();
    }

    // Step 2: Binary search to find f_min
    double epsilon = 0.01;
    double f_low = max(h[AX][AY], h[BX][BY]);  // Start from the max elevation between A and B
    double f_high = max_h;
    double f_min = max_h;

    while (f_high - f_low > epsilon) {
        double f_mid = (f_low + f_high) / 2.0;

        // Check if both starting and ending points are below f_mid
        if (h[AX][AY] <= f_mid && h[BX][BY] <= f_mid) {
            // A* algorithm to check for path at elevation f_mid
            vector<vector<double>> dist(GRID_SIZE, vector<double>(GRID_SIZE, numeric_limits<double>::infinity()));
            priority_queue<pair<double, pair<int, int>>, vector<pair<double, pair<int, int>>>, greater<>> pq;

            dist[AX][AY] = 0.0;
            pq.push({heuristic(AX, AY), {AX, AY}});

            bool path_found = false;

            while (!pq.empty()) {
                auto [current_f, point] = pq.top();
                pq.pop();
                int xi = point.first;
                int yi = point.second;

                double current_dist = current_f - heuristic(xi, yi);

                if (current_dist > dist[xi][yi]) continue;

                if (xi == BX && yi == BY) {
                    path_found = true;
                    break;
                }

                for (int dir = 0; dir < 8; ++dir) {
                    int nx = xi + dx[dir];
                    int ny = yi + dy[dir];
                    double edge_dist = move_cost[dir];
                    if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE &&
                        h[nx][ny] <= f_mid) {
                        double new_dist = dist[xi][yi] + edge_dist;
                        if (new_dist < dist[nx][ny]) {
                            dist[nx][ny] = new_dist;
                            pq.push({new_dist + heuristic(nx, ny), {nx, ny}});
                        }
                    }
                }
            }

            if (path_found) {
                f_min = f_mid;
                f_high = f_mid;
            } else {
                f_low = f_mid;
            }
        } else {
            f_low = f_mid;
        }
    }

    // Step 3: Compute the shortest path at elevation f_min using A*
    // Ensure that both starting and ending points are below f_min
    if (h[AX][AY] <= f_min && h[BX][BY] <= f_min) {
        vector<vector<double>> dist(GRID_SIZE, vector<double>(GRID_SIZE, numeric_limits<double>::infinity()));
        priority_queue<pair<double, pair<int, int>>, vector<pair<double, pair<int, int>>>, greater<>> pq;

        dist[AX][AY] = 0.0;
        pq.push({heuristic(AX, AY), {AX, AY}});

        while (!pq.empty()) {
            auto [current_f, point] = pq.top();
            pq.pop();
            int xi = point.first;
            int yi = point.second;

            double current_dist = current_f - heuristic(xi, yi);

            if (current_dist > dist[xi][yi]) continue;

            if (xi == BX && yi == BY) {
                break;
            }

            for (int dir = 0; dir < 8; ++dir) {
                int nx = xi + dx[dir];
                int ny = yi + dy[dir];
                double edge_dist = move_cost[dir];
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE &&
                    h[nx][ny] <= f_min) {
                    double new_dist = dist[xi][yi] + edge_dist;
                    if (new_dist < dist[nx][ny]) {
                        dist[nx][ny] = new_dist;
                        pq.push({new_dist + heuristic(nx, ny), {nx, ny}});
                    }
                }
            }
        }

        double shortest_path_length = dist[BX][BY];

        if (isinf(shortest_path_length)) {
            cout << "No path found under the elevation f_min." << endl;
        } else {
            // Stop measuring time
            auto end_time = high_resolution_clock::now();
            duration<double> elapsed = end_time - start_time;

            cout << fixed << setprecision(3);
            cout << "Minimum elevation (f_min): " << f_min << endl;
            cout << "Length of the shortest path: " << shortest_path_length << endl;
            cout << "Execution time: " << elapsed.count() << " seconds" << endl;
        }
    } else {
        cout << "Cannot find path under elevation f_min." << endl;
    }

    return 0;
}