from ortools.sat.python import cp_model

def find_all_solutions():
    # Create the model
    model = cp_model.CpModel()

    # Define variables for A
    a1 = model.NewIntVar(1, 9, 'a1')  # Hundreds place of A
    a2 = model.NewIntVar(0, 9, 'a2')  # Tens place of A
    a3 = model.NewIntVar(0, 9, 'a3')  # Units place of A

    # Define variables for B
    b1 = model.NewIntVar(1, 9, 'b1')  # Hundreds place of B
    b2 = model.NewIntVar(0, 9, 'b2')  # Tens place of B
    b3 = model.NewIntVar(0, 9, 'b3')  # Units place of B

    # Define the difference constraint: A - B = 234
    model.Add(100 * a1 + 10 * a2 + a3 - (100 * b1 + 10 * b2 + b3) == 234)

    # All digits in A and B must be unique and cannot include 2, 3, or 4
    digits = [a1, a2, a3, b1, b2, b3]
    model.AddAllDifferent(digits)

    # Exclude digits 2, 3, 4 from A and B
    forbidden_digits = [2, 3, 4]
    for d in digits:
        for fd in forbidden_digits:
            model.Add(d != fd)

    # Create a solver
    solver = cp_model.CpSolver()

    # Container to store all solutions
    solutions = []

    # Custom callback to collect solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.solutions = []

        def on_solution_callback(self):
            A_val = self.Value(a1) * 100 + self.Value(a2) * 10 + self.Value(a3)
            B_val = self.Value(b1) * 100 + self.Value(b2) * 10 + self.Value(b3)
            sum_val = A_val + B_val
            self.solutions.append({
                'A': A_val,
                'B': B_val,
                'A_minus_B': A_val - B_val,
                'A_plus_B': sum_val,
                'Digits_A': (self.Value(a1), self.Value(a2), self.Value(a3)),
                'Digits_B': (self.Value(b1), self.Value(b2), self.Value(b3))
            })

    # Initialize the solution collector
    collector = SolutionCollector()

    # Solve the model and collect all solutions
    status = solver.SearchForAllSolutions(model, collector)

    # Check if any solutions were found
    if collector.solutions:
        # Sort the solutions based on A + B in descending order
        sorted_solutions = sorted(collector.solutions, key=lambda x: x['A_plus_B'], reverse=True)

        print(f"Total Solutions Found: {len(sorted_solutions)}\n")
        print("Solutions (sorted by A + B descending):\n")
        for idx, sol in enumerate(sorted_solutions, start=1):
            print(f"Solution {idx}:")
            print(f"  A = {sol['A']}")
            print(f"  B = {sol['B']}")
            print(f"  A - B = {sol['A_minus_B']}")
            print(f"  A + B = {sol['A_plus_B']}")
            print(f"  Digits in A: {sol['Digits_A']}")
            print(f"  Digits in B: {sol['Digits_B']}")
            print(f"  Difference (234) Digits: [2, 3, 4]\n")
    else:
        print("No solutions found.")

if __name__ == "__main__":
    find_all_solutions()