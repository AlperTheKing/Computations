from ortools.sat.python import cp_model

def solve_no_three_in_line(n=5):
    """
    Solves the "no-three-in-line" problem on an n x n grid using OR-Tools.
    Returns the maximum number of pawns that can be placed without any three in a line,
    and one such arrangement.
    """
    model = cp_model.CpModel()

    # Create variables
    pawns = {}
    for i in range(n):
        for j in range(n):
            pawns[(i, j)] = model.NewBoolVar(f'pawn_{i}_{j}')

    # Constraints: No three pawns in any horizontal line
    for i in range(n):
        model.Add(sum(pawns[(i, j)] for j in range(n)) <= 2)

    # Constraints: No three pawns in any vertical line
    for j in range(n):
        model.Add(sum(pawns[(i, j)] for i in range(n)) <= 2)

    # Constraints: No three pawns in any diagonal (both main and anti-diagonals)
    # Main diagonals: i - j = constant
    for k in range(-n + 1, n):
        diagonal = [pawns[(i, j)] for i in range(n) for j in range(n) if i - j == k]
        if len(diagonal) >= 3:
            model.Add(sum(diagonal) <= 2)

    # Anti-diagonals: i + j = constant
    for k in range(2 * n - 1):
        anti_diagonal = [pawns[(i, j)] for i in range(n) for j in range(n) if i + j == k]
        if len(anti_diagonal) >= 3:
            model.Add(sum(anti_diagonal) <= 2)

    # Objective: Maximize the number of pawns
    model.Maximize(sum(pawns[(i, j)] for i in range(n) for j in range(n)))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        max_pawns = int(solver.ObjectiveValue())
        arrangement = [['.' for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if solver.Value(pawns[(i, j)]) == 1:
                    arrangement[i][j] = 'P'
        return max_pawns, arrangement
    else:
        return None, None

def print_arrangement(arrangement):
    """
    Prints the chessboard arrangement.
    """
    if arrangement is None:
        print("No solution found.")
    else:
        print("Chessboard Arrangement:")
        for row in arrangement:
            print(' '.join(row))

def main():
    n = 5  # Size of the chessboard
    max_pawns, arrangement = solve_no_three_in_line(n)
    if max_pawns is not None:
        print(f"Maximum number of pawns without any three in a line: {max_pawns}")
        print_arrangement(arrangement)
        print(f"\nMinimum X to guarantee at least one line with three pawns: {max_pawns + 1}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()