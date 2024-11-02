# File: abc_connection_grid_output.py

from ortools.sat.python import cp_model

def main():
    # Create the model
    model = cp_model.CpModel()

    # Define the grid size
    N = 10
    cells = range(N)

    # Define the letters and their positions (0-based indexing)
    letters = {
        1: [(5, 4), (9, 0)],  # 'A'
        2: [(0, 0), (2, 7)],  # 'B'
        3: [(1, 1), (0, 4)],  # 'C'
        4: [(1, 8), (8, 0)],  # 'D'
        5: [(6, 6), (9, 2)],  # 'E'
        6: [(4, 5), (7, 4)],  # 'F'
        7: [(3, 7), (7, 3)],  # 'G'
        8: [(4, 1), (8, 4)],  # 'H'
    }

    num_letters = len(letters)

    # Create variables for each cell in the grid
    grid = {}
    for i in cells:
        for j in cells:
            grid[(i, j)] = model.NewIntVar(0, num_letters, f'grid_{i}_{j}')

    # Define possible moves (up, down, left, right)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Create is_in_path variables for all letters
    is_in_path = {}
    for letter in letters.keys():
        is_in_path[letter] = {}
        for i in cells:
            for j in cells:
                is_in_path[letter][(i, j)] = model.NewBoolVar(f'is_in_path_{letter}_{i}_{j}')

    # For each letter, set up constraints
    for letter, positions in letters.items():
        start = positions[0]
        end = positions[1]

        # Enforce that start and end cells are part of the path
        model.Add(is_in_path[letter][start] == 1)
        model.Add(is_in_path[letter][end] == 1)

        # Enforce that if a cell is part of the path, grid cell value equals the letter
        for i in cells:
            for j in cells:
                # Link grid cell values and is_in_path variables
                model.Add(grid[(i, j)] == letter).OnlyEnforceIf(is_in_path[letter][(i, j)])
                model.Add(grid[(i, j)] != letter).OnlyEnforceIf(is_in_path[letter][(i, j)].Not())

        # Path continuity constraints
        for i in cells:
            for j in cells:
                if (i, j) != start and (i, j) != end:
                    # If the cell is part of the path, it must have exactly two neighbors in the path
                    neighbors = []
                    for move in moves:
                        ni, nj = i + move[0], j + move[1]
                        if 0 <= ni < N and 0 <= nj < N:
                            neighbors.append(is_in_path[letter][(ni, nj)])
                    model.Add(sum(neighbors) == 2).OnlyEnforceIf(is_in_path[letter][(i, j)])
                    # If the cell is not part of the path, no constraint needed

        # For start and end cells, they must have exactly one neighbor in the path
        for pos in [start, end]:
            i, j = pos
            neighbors = []
            for move in moves:
                ni, nj = i + move[0], j + move[1]
                if 0 <= ni < N and 0 <= nj < N:
                    neighbors.append(is_in_path[letter][(ni, nj)])
            model.Add(sum(neighbors) == 1)

    # Ensure that no cell is used by more than one letter
    for i in cells:
        for j in cells:
            letter_in_cell = []
            for letter in letters.keys():
                letter_in_cell.append(is_in_path[letter][(i, j)])
            model.Add(sum(letter_in_cell) <= 1)

    # Set the starting and ending positions for each letter (ensure consistency)
    for letter, positions in letters.items():
        for pos in positions:
            model.Add(grid[pos] == letter)

    # Create the solver and solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0  # Set a time limit

    # Solve the model
    status = solver.Solve(model)

    # Output the solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create a mapping from letter numbers to actual letters
        letter_map = {i: chr(ord('A') + i - 1) for i in range(1, num_letters + 1)}

        print("\nSolution Found:")
        # Print column indices
        header = '   ' + ' '.join(f'{j:2}' for j in cells)
        print(header)
        print('  +' + '--' * N)
        for i in cells:
            row = f'{i:2}|'
            for j in cells:
                val = int(solver.Value(grid[(i, j)]))
                if val == 0:
                    row += '. '
                else:
                    row += f'{letter_map[val]} '
            print(row)
    else:
        print("\nNo solution found.")

if __name__ == '__main__':
    main()