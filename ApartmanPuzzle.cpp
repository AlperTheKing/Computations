#include <iostream>
#include <vector>
#include <set>
#include <chrono>

const int grid_size = 5;
std::vector<std::vector<int>> puzzle(grid_size, std::vector<int>(grid_size, 0));

// Clues for the number of visible buildings from each side (None represented by -1)
std::vector<int> top_clues    = { 3, -1, -1, -1, 3 };
std::vector<int> bottom_clues = { -1, -1, -1, -1, 3 };
std::vector<int> left_clues   = { -1, -1, 3, -1, 2 };
std::vector<int> right_clues  = { 4, -1, -1, -1, 2 };

// Define the bolded regions, each region has unique numbers 1-5
std::vector<std::vector<std::pair<int, int>>> regions = {
    { {0, 0}, {0, 1}, {1, 1}, {1, 2}, {2, 2} },  // Top-left region
    { {1, 0}, {2, 0}, {3, 0}, {4, 0}, {2, 1} },  // Top-middle region
    { {0, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 4} },  // Top-right region
    { {1, 3}, {2, 3}, {3, 3}, {3, 1}, {3, 2} },  // Bottom-left region
    { {3, 4}, {4, 1}, {4, 2}, {4, 3}, {4, 4} }   // Bottom-right region
};

// Store all possible solutions
std::vector<std::vector<std::vector<int>>> solutions;

bool is_valid(const std::vector<std::vector<int>>& grid, int row, int col, int num) {
    // Check the row and column for uniqueness
    for (int i = 0; i < grid_size; ++i) {
        if (grid[row][i] == num || grid[i][col] == num) {
            return false;
        }
    }
    return true;
}

int count_visible_buildings(const std::vector<int>& arr) {
    int max_height = 0;
    int visible_count = 0;
    for (int height : arr) {
        if (height > max_height) {
            ++visible_count;
            max_height = height;
        }
    }
    return visible_count;
}

bool check_regions(const std::vector<std::vector<int>>& grid) {
    for (const auto& region : regions) {
        std::set<int> seen;
        for (const auto& [r, c] : region) {
            int num = grid[r][c];
            if (num != 0) {
                if (seen.count(num)) {
                    return false;
                }
                seen.insert(num);
            }
        }
    }
    return true;
}

bool check_clues(const std::vector<std::vector<int>>& grid) {
    for (int i = 0; i < grid_size; ++i) {
        // Check top clues
        if (top_clues[i] != -1) {
            std::vector<int> column(grid_size);
            for (int j = 0; j < grid_size; ++j) {
                column[j] = grid[j][i];
            }
            if (count_visible_buildings(column) != top_clues[i]) return false;
        }
        // Check bottom clues
        if (bottom_clues[i] != -1) {
            std::vector<int> column(grid_size);
            for (int j = 0; j < grid_size; ++j) {
                column[j] = grid[grid_size - 1 - j][i];
            }
            if (count_visible_buildings(column) != bottom_clues[i]) return false;
        }
        // Check left clues
        if (left_clues[i] != -1 && count_visible_buildings(grid[i]) != left_clues[i]) {
            return false;
        }
        // Check right clues
        if (right_clues[i] != -1) {
            std::vector<int> row = grid[i];
            std::reverse(row.begin(), row.end());
            if (count_visible_buildings(row) != right_clues[i]) return false;
        }
    }
    return true;
}

void solve(std::vector<std::vector<int>>& grid, int row = 0, int col = 0) {
    if (row == grid_size) {
        if (check_clues(grid) && check_regions(grid)) {
            // Store the valid solution
            solutions.push_back(grid);
        }
        return;
    }
    
    if (col == grid_size) {
        solve(grid, row + 1, 0);
        return;
    }
    
    if (grid[row][col] != 0) {
        solve(grid, row, col + 1);
        return;
    }
    
    for (int num = 1; num <= grid_size; ++num) {
        if (is_valid(grid, row, col, num)) {
            grid[row][col] = num;
            solve(grid, row, col + 1);
            grid[row][col] = 0;  // Reset for next possibility
        }
    }
}

int main() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Solve the puzzle
    solve(puzzle);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    // Output the results
    std::cout << "Time taken: " << duration.count() << " seconds\n";
    
    if (!solutions.empty()) {
        std::cout << "Number of solutions found: " << solutions.size() << "\n";
        for (size_t idx = 0; idx < solutions.size(); ++idx) {
            std::cout << "Solution " << idx + 1 << ":\n";
            for (const auto& row : solutions[idx]) {
                for (int num : row) {
                    std::cout << num << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "No solutions found.\n";
    }

    return 0;
}