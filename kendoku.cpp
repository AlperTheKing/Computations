#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

const int GRID_SIZE = 5;  // 5x5 grid for Kendoku
std::vector<std::vector<int>> grid(GRID_SIZE, std::vector<int>(GRID_SIZE, 0));
int solution_count = 0; // Counter for the number of solutions

// Region structure now includes possible operations
struct Region {
    std::vector<std::pair<int, int>> cells;
    int target;
    std::vector<char> operations = {'+', '-', '*', '/'};  // Possible operations for each region
};

// Initialize regions here based on the puzzle image (fill with cells and targets)
std::vector<Region> regions = {
    {{ {0, 0}, {0, 1} }, 4},
    {{ {0, 2}, {1, 2} }, 5},
    {{ {0, 3}, {0, 4} }, 15},
    {{ {1, 0}, {1, 1} }, 5},
    {{ {1, 3}, {1, 4} }, 2},
    {{ {2, 0}, {2, 1} }, 3},
    {{ {2, 2}, {2, 3} }, 7},
    {{ {2, 4}, {3, 4} }, 5},
    {{ {3, 0}, {4, 0} }, 7},
    {{ {3, 1}, {4, 1} }, 5},
    {{ {3, 2}, {3, 3}, {4,2} }, 20},
    {{ {4, 3}, {4, 4} }, 2}
};

// Function to check if the placement of a number is valid in the row and column
bool isValidPlacement(int row, int col, int num) {
    for (int i = 0; i < GRID_SIZE; i++) {
        if (grid[row][i] == num || grid[i][col] == num) {
            return false;
        }
    }
    return true;
}

// Evaluate the region with all possible operations, returning true if one matches the target
bool checkRegion(const Region &region) {
    for (char op : region.operations) {
        int result = (op == '+') ? 0 : (op == '*') ? 1 : grid[region.cells[0].first][region.cells[0].second];
        bool valid = true;
        
        for (size_t i = (op == '-' || op == '/') ? 1 : 0; i < region.cells.size(); ++i) {
            int value = grid[region.cells[i].first][region.cells[i].second];
            if (value == 0) { valid = false; break; }  // Skip if any cell is unfilled
            
            switch(op) {
                case '+': result += value; break;
                case '*': result *= value; break;
                case '-': 
                    result = std::abs(result - value); 
                    break;
                case '/': 
                    if (value == 0 || result % value != 0) { valid = false; }
                    else result /= value;
                    break;
            }
        }
        if (valid && result == region.target) return true;
    }
    return false;
}

// Check if the current grid setup satisfies all regions
bool allRegionsValid() {
    for (const auto &region : regions) {
        if (!checkRegion(region)) return false;
    }
    return true;
}

// Function to print the grid (solution)
void printGrid() {
    for (const auto &row : grid) {
        for (int num : row) {
            std::cout << num << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Backtracking function to find all solutions
void solveAll(int row, int col) {
    if (row == GRID_SIZE) {
        if (allRegionsValid()) {
            solution_count++;
            std::cout << "Solution #" << solution_count << ":\n";
            printGrid();
        }
        return;
    }
    if (col == GRID_SIZE) {
        solveAll(row + 1, 0);
        return;
    }

    for (int num = 1; num <= GRID_SIZE; num++) {
        if (isValidPlacement(row, col, num)) {
            grid[row][col] = num;
            solveAll(row, col + 1);
            grid[row][col] = 0; // Backtrack
        }
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    solveAll(0, 0);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total solutions found: " << solution_count << "\n";
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}