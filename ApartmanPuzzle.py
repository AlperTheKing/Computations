import numpy as np
import time  # Import the time module to measure execution time

# Define the puzzle grid and clues
grid_size = 5
puzzle = np.zeros((grid_size, grid_size), dtype=int)

# Clues for the number of visible buildings from each side (None represents no clue)
top_clues = [None, None, 4, None, None]
bottom_clues = [None, None, None, None, None]
left_clues = [None, None, None, 4, None]
right_clues = [2, None, None, None, None]

# Define the bolded regions, each region has unique numbers 1-5
regions = [
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],  # Top-left region
    [(3, 0), (4, 0), (4, 1), (4, 2), (3, 2)],  # Top-middle region
    [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3)],  # Top-right region
    [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],  # Bottom-left region
    [(1, 4), (2, 4), (3, 4), (4, 4), (4, 3)]   # Bottom-right region
]

# Store all possible solutions
solutions = []

# Function to check if a number can be placed at a given row, col
def is_valid(grid, row, col, num):
    # Check the row and column for uniqueness
    if num in grid[row, :] or num in grid[:, col]:
        return False
    return True

# Function to count the visible buildings from a row or column
def count_visible_buildings(arr):
    max_height = 0
    visible_count = 0
    for height in arr:
        if height > max_height:
            visible_count += 1
            max_height = height
    return visible_count

# Function to check if the numbers in each bolded region are unique
def check_regions(grid):
    for region in regions:
        seen = set()
        for r, c in region:
            num = grid[r, c]
            if num != 0:
                if num in seen:
                    return False
                seen.add(num)
    return True

# Function to check if current grid state is valid according to the clues
def check_clues(grid):
    for i in range(grid_size):
        # Check top clues
        if top_clues[i] is not None and count_visible_buildings(grid[:, i]) != top_clues[i]:
            return False
        # Check bottom clues
        if bottom_clues[i] is not None and count_visible_buildings(grid[:, i][::-1]) != bottom_clues[i]:
            return False
        # Check left clues
        if left_clues[i] is not None and count_visible_buildings(grid[i, :]) != left_clues[i]:
            return False
        # Check right clues
        if right_clues[i] is not None and count_visible_buildings(grid[i, :][::-1]) != right_clues[i]:
            return False
    return True

# Backtracking function to solve the puzzle
def solve(grid, row=0, col=0):
    if row == grid_size:
        if check_clues(grid) and check_regions(grid):
            # Store the valid solution
            solutions.append(np.copy(grid))
        return
    
    if col == grid_size:
        solve(grid, row + 1, 0)
        return
    
    if grid[row, col] != 0:
        solve(grid, row, col + 1)
        return
    
    for num in range(1, grid_size + 1):
        if is_valid(grid, row, col, num):
            grid[row, col] = num
            solve(grid, row, col + 1)
            grid[row, col] = 0  # Reset for next possibility

# Main function to initialize the puzzle and solve it
def main():
    start_time = time.time()  # Start timing
    solve(puzzle)
    end_time = time.time()  # End timing
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    if solutions:
        print(f"Number of solutions found: {len(solutions)}\n")
        for idx, solution in enumerate(solutions):
            print(f"Solution {idx + 1}:\n{solution}\n")
    else:
        print("No solutions found")
    
    print(f"Time taken: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()