from ortools.sat.python import cp_model

def find_all_solutions():
    model = cp_model.CpModel()

    # Define variables for each digit in the three numbers
    # A = A1 A2 A3
    # B = B1 B2 B3
    # C = C1 C2 C3

    # Hundreds digits: 1-9 (cannot be 0)
    A1 = model.NewIntVar(1, 9, 'A1')
    B1 = model.NewIntVar(1, 9, 'B1')
    C1 = model.NewIntVar(1, 9, 'C1')

    # Tens and units digits: 0-9
    A2 = model.NewIntVar(0, 9, 'A2')
    A3 = model.NewIntVar(0, 9, 'A3')

    B2 = model.NewIntVar(0, 9, 'B2')
    B3 = model.NewIntVar(0, 9, 'B3')

    C2 = model.NewIntVar(0, 9, 'C2')
    C3 = model.NewIntVar(0, 9, 'C3')

    # Collect all digit variables
    digits = [A1, A2, A3, B1, B2, B3, C1, C2, C3]

    # All digits must be distinct
    model.AddAllDifferent(digits)

    # Define the numbers
    A = model.NewIntVar(100, 999, 'A')
    B = model.NewIntVar(100, 999, 'B')
    C = model.NewIntVar(100, 999, 'C')

    model.Add(A == 100 * A1 + 10 * A2 + A3)
    model.Add(B == 100 * B1 + 10 * B2 + B3)
    model.Add(C == 100 * C1 + 10 * C2 + C3)

    # Enforce ascending order
    model.Add(A < B)
    model.Add(B < C)

    # Sum constraint
    model.Add(A + B + C == 1999)

    # Create a solver
    solver = cp_model.CpSolver()

    # Define a callback to collect all solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, A, B, C):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.A = A
            self.B = B
            self.C = C
            self.solutions = []

        def on_solution_callback(self):
            A_val = self.Value(self.A)
            B_val = self.Value(self.B)
            C_val = self.Value(self.C)
            self.solutions.append((A_val, B_val, C_val))

    collector = SolutionCollector(A, B, C)

    # Search for all solutions
    solver.SearchForAllSolutions(model, collector)

    # Process and display the solutions
    if collector.solutions:
        print(f"Total Solutions Found: {len(collector.solutions)}\n")
        for idx, sol in enumerate(collector.solutions, 1):
            A_val, B_val, C_val = sol
            print(f"Solution {idx}:")
            print(f"  A = {A_val}")
            print(f"  B = {B_val}")
            print(f"  C = {C_val}")
            print(f"  Sum = {A_val + B_val + C_val}\n")

        # Find the solution with the minimum B
        min_B_solution = min(collector.solutions, key=lambda x: x[1])
        print(f"Minimum B found: {min_B_solution[1]}")
        print(f"Corresponding Solution:")
        print(f"  A = {min_B_solution[0]}")
        print(f"  B = {min_B_solution[1]}")
        print(f"  C = {min_B_solution[2]}")
        print(f"  Sum = {min_B_solution[0] + min_B_solution[1] + min_B_solution[2]}")
    else:
        print("No solution exists under the given constraints.")

if __name__ == "__main__":
    find_all_solutions()
