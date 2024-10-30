// Filename: mosquito_path.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <limits>
#include <chrono>
#include <omp.h>

// Constants
const int GRID_SIZE = 20000;  // Adjusted grid size (increase by 10x for practicality)
const double X_MIN = 0.0;
const double X_MAX = 1600.0;
const double Y_MIN = 0.0;
const double Y_MAX = 1600.0;

// Elevation function h(x, y)
double h(double x, double y) {
    double term1 = 5000.0 - 0.005 * (x * x + y * y + x * y) + 12.5 * (x + y);
    double term2 = -std::abs(0.000001 * (x * x + y * y) - 0.0015 * (x + y) + 0.7);
    double h_xy = term1 * std::exp(term2);
    return h_xy;
}

// Mapping from (x, y) to grid indices
inline int point_to_index(double coord, double min_val, double delta) {
    return static_cast<int>(std::round((coord - min_val) / delta));
}

// Mapping from grid indices to (x, y)
inline double index_to_point(int index, double min_val, double delta) {
    return min_val + index * delta;
}

// Neighboring positions (including diagonals)
const std::vector<std::pair<int, int>> neighbors = {
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1},          { 0, 1},
    { 1, -1}, { 1, 0}, { 1, 1}
};

// Function to check if A and B are connected at a given elevation threshold
bool is_reachable(const std::vector<std::vector<double>>& H, int start_i, int start_j, int end_i, int end_j, double f_min) {
    int grid_size = H.size();
    std::vector<std::vector<bool>> visited(grid_size, std::vector<bool>(grid_size, false));

    std::queue<std::pair<int, int>> queue;
    queue.push({start_i, start_j});
    visited[start_i][start_j] = true;

    while (!queue.empty()) {
        auto [i, j] = queue.front();
        queue.pop();
        if (i == end_i && j == end_j) {
            return true;
        }
        for (const auto& [di, dj] : neighbors) {
            int ni = i + di;
            int nj = j + dj;
            if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                if (!visited[ni][nj] && H[ni][nj] <= f_min) {
                    visited[ni][nj] = true;
                    queue.push({ni, nj});
                }
            }
        }
    }
    return false;
}

// A* algorithm to find the shortest path at elevation f_min
double a_star_shortest_path(const std::vector<std::vector<double>>& H, int start_i, int start_j, int end_i, int end_j,
                            double dx, double dy, double f_min) {
    int grid_size = H.size();
    std::vector<std::vector<bool>> visited(grid_size, std::vector<bool>(grid_size, false));
    std::vector<std::vector<double>> distance(grid_size, std::vector<double>(grid_size, std::numeric_limits<double>::infinity()));

    auto compare = [](const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) {
        return std::get<0>(a) > std::get<0>(b);  // Min-heap based on f-score
    };
    std::priority_queue<std::tuple<double, int, int>, std::vector<std::tuple<double, int, int>>, decltype(compare)> heap(compare);

    distance[start_i][start_j] = 0.0;
    double heuristic = std::hypot((start_j - end_j) * dx, (start_i - end_i) * dy);
    heap.push({heuristic, start_i, start_j});

    while (!heap.empty()) {
        auto [curr_f, i, j] = heap.top();
        heap.pop();
        if (visited[i][j]) continue;
        if (i == end_i && j == end_j) break;
        visited[i][j] = true;
        for (const auto& [di, dj] : neighbors) {
            int ni = i + di;
            int nj = j + dj;
            if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                if (!visited[ni][nj] && H[ni][nj] <= f_min) {
                    double x1 = index_to_point(j, X_MIN, dx);
                    double y1 = index_to_point(i, Y_MIN, dy);
                    double x2 = index_to_point(nj, X_MIN, dx);
                    double y2 = index_to_point(ni, Y_MIN, dy);
                    double dist = std::hypot(x2 - x1, y2 - y1);
                    double tentative_g = distance[i][j] + dist;
                    if (tentative_g < distance[ni][nj]) {
                        distance[ni][nj] = tentative_g;
                        double heuristic = std::hypot(x2 - (X_MIN + (end_j * dx)), y2 - (Y_MIN + (end_i * dy)));
                        double f = tentative_g + heuristic;
                        heap.push({f, ni, nj});
                    }
                }
            }
        }
    }
    return distance[end_i][end_j];
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set the number of threads to use all available system threads
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    std::cout << "Using " << max_threads << " threads." << std::endl;

    // Grid setup
    double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
    double dy = (Y_MAX - Y_MIN) / (GRID_SIZE - 1);

    // Generate grid points
    std::vector<double> x_vals(GRID_SIZE);
    std::vector<double> y_vals(GRID_SIZE);
    for (int i = 0; i < GRID_SIZE; ++i) {
        x_vals[i] = X_MIN + i * dx;
        y_vals[i] = Y_MIN + i * dy;
    }

    // Elevation grid H
    std::vector<std::vector<double>> H(GRID_SIZE, std::vector<double>(GRID_SIZE));

    // Parallel computation of H using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            double x = x_vals[j];
            double y = y_vals[i];
            H[i][j] = h(x, y);
        }
    }

    // Starting and ending points
    int start_i = point_to_index(200.0, Y_MIN, dy);
    int start_j = point_to_index(200.0, X_MIN, dx);
    int end_i = point_to_index(1400.0, Y_MIN, dy);
    int end_j = point_to_index(1400.0, X_MIN, dx);

    // Binary search to find f_min
    std::vector<double> elevations;
    elevations.reserve(GRID_SIZE * GRID_SIZE);

    // Flatten the elevation grid and collect elevations
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < GRID_SIZE; ++i) {
        std::vector<double> local_elevations;
        for (int j = 0; j < GRID_SIZE; ++j) {
            local_elevations.push_back(H[i][j]);
        }
        #pragma omp critical
        elevations.insert(elevations.end(), local_elevations.begin(), local_elevations.end());
    }

    // Remove duplicates and sort
    std::sort(elevations.begin(), elevations.end());
    auto last = std::unique(elevations.begin(), elevations.end());
    elevations.erase(last, elevations.end());

    size_t left = 0;
    size_t right = elevations.size() - 1;
    double f_min = 0.0;

    while (left <= right) {
        size_t mid = (left + right) / 2;
        double elevation_threshold = elevations[mid];
        if (is_reachable(H, start_i, start_j, end_i, end_j, elevation_threshold)) {
            f_min = elevation_threshold;
            if (mid == 0) break;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    std::cout << "Minimum elevation f_min: " << std::fixed << std::setprecision(3) << f_min << std::endl;

    // Compute the shortest path length using A* algorithm
    double path_length = a_star_shortest_path(H, start_i, start_j, end_i, end_j, dx, dy, f_min);
    std::cout << "Length of the shortest path: " << std::fixed << std::setprecision(3) << path_length << std::endl;

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}