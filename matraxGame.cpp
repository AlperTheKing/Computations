#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const int N = 5;  // Grid size

// Clue structure to represent the given clues
struct Clue {
    int row1, col1;  // First cell in the diagonal pair (top-right to bottom-left)
    int row2, col2;  // Second cell in the diagonal pair (top-right to bottom-left)
    char operation;  // The operation ('+', '-', 'x', '/')
    int value;       // The target value for the operation
};

// Function to check if a clue is satisfied for both diagonals
bool check_clue(int num1, int num2, int num3, int num4, const Clue& clue) {
    bool first_diagonal = false, second_diagonal = false;

    // First diagonal pair (entered pair)
    switch (clue.operation) {
        case '+':
            first_diagonal = (num1 + num2 == clue.value);
            break;
        case '-':
            first_diagonal = (abs(num1 - num2) == clue.value);
            break;
        case 'x':
            first_diagonal = (num1 * num2 == clue.value);
            break;
        case '/':
            first_diagonal = (num1 / num2 == clue.value || num2 / num1 == clue.value);
            break;
        default:
            cout << "Unsupported operation: " << clue.operation << endl;
            return false;
    }

    // Second diagonal pair (the other diagonal)
    switch (clue.operation) {
        case '+':
            second_diagonal = (num3 + num4 == clue.value);
            break;
        case '-':
            second_diagonal = (abs(num3 - num4) == clue.value);
            break;
        case 'x':
            second_diagonal = (num3 * num4 == clue.value);
            break;
        case '/':
            second_diagonal = (num3 / num4 == clue.value || num4 / num3 == clue.value);
            break;
    }

    return first_diagonal && second_diagonal;
}

// Function to check if all clues are satisfied for the current grid
bool check_clues(const vector<vector<int>>& grid, const vector<Clue>& clues) {
    for (const Clue& clue : clues) {
        int num1 = grid[clue.row1][clue.col1];  // First number from clue
        int num2 = grid[clue.row2][clue.col2];  // Second number from clue
        int num3 = grid[clue.row1][clue.col2];  // Top-left number for the second diagonal
        int num4 = grid[clue.row2][clue.col1];  // Bottom-right number for the second diagonal

        if (!check_clue(num1, num2, num3, num4, clue)) {
            return false;
        }
    }
    return true;
}

// Function to check if a number is valid to be placed in a cell (no duplicates in rows or columns)
bool is_valid(const vector<vector<int>>& grid, int row, int col, int num) {
    for (int i = 0; i < N; ++i) {
        if (grid[row][i] == num || grid[i][col] == num) return false;
    }
    return true;
}

// Backtracking function to solve the grid and find all solutions
void solve(vector<vector<int>>& grid, const vector<Clue>& clues, int row, int col, vector<vector<vector<int>>>& solutions) {
    if (row == N) {
        if (check_clues(grid, clues)) {
            solutions.push_back(grid);  // Store the solution
        }
        return;  // Continue to find other solutions
    }

    int next_row = (col == N - 1) ? row + 1 : row;
    int next_col = (col == N - 1) ? 0 : col + 1;

    for (int num = 1; num <= N; ++num) {
        if (is_valid(grid, row, col, num)) {
            grid[row][col] = num;  // Place the number in the grid
            solve(grid, clues, next_row, next_col, solutions);  // Continue to solve
            grid[row][col] = 0;  // Backtrack and remove the number
        }
    }
}

int main() {
    // Initialize an empty grid
    vector<vector<int>> grid(N, vector<int>(N, 0));

    // Define the clues
    vector<Clue> clues = {
        {0, 2, 1, 1, '-', 1},  // Clue for 1-
        {1, 1, 2, 0, '-', 1},  // Clue for 7+
        {1, 3, 2, 2, '-', 1},  // Clue for 4+
        {2, 2, 3, 1, '-', 1},  // Clue for 6+
        {2, 3, 3, 2, '+', 6},  // Clue for 1-
        {3, 4, 4, 3, 'x', 4},  // Clue for 3-
    };

    // Vector to store all solutions
    vector<vector<vector<int>>> solutions;

    // Solve the puzzle
    solve(grid, clues, 0, 0, solutions);

    // Output all solutions
    if (!solutions.empty()) {
        cout << "Found " << solutions.size() << " solutions:" << endl;
        for (const auto& sol : solutions) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    cout << sol[i][j] << " ";
                }
                cout << endl;
            }
            cout << "---------" << endl;
        }
    } else {
        cout << "No solution found." << endl;
    }

    return 0;
}