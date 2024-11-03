from ortools.sat.python import cp_model
import time

# Define the grid dimensions and initial grid with numbers
GRID_SIZE = 9
initial_grid = [
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 6, 16, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 12, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 12, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0]
]

# Define the areas for each rectangle based on the initial grid numbers
area_constraints = {
    (0, 1): 10,
    (0, 4): 5,
    (2, 6): 6,
    (2, 7): 16,
    (4, 4): 12,
    (6, 1): 12,
    (6, 2): 3,
    (7, 5): 9,
    (8, 6): 3
}

# Define the model
model = cp_model.CpModel()

# Create variables for the coordinates and dimensions of each rectangle
rectangles = {}
for (x, y), area in area_constraints.items():
    rectangles[(x, y)] = {
        'x': model.NewIntVar(0, GRID_SIZE - 1, f"x_{x}_{y}"),
        'y': model.NewIntVar(0, GRID_SIZE - 1, f"y_{x}_{y}"),
        'width': model.NewIntVar(1, area, f"width_{x}_{y}"),
        'height': model.NewIntVar(1, area, f"height_{x}_{y}")
    }

# Constraints for each rectangle to meet the area requirement
for (x, y), area in area_constraints.items():
    rect = rectangles[(x, y)]
    model.Add(rect['width'] * rect['height'] == area)

    # Keep rectangles within grid bounds
    model.Add(rect['x'] + rect['width'] <= GRID_SIZE)
    model.Add(rect['y'] + rect['height'] <= GRID_SIZE)

# Ensure rectangles do not overlap
for (x1, y1), rect1 in rectangles.items():
    for (x2, y2), rect2 in rectangles.items():
        if (x1, y1) != (x2, y2):
            model.AddBoolOr([
                rect1['x'] + rect1['width'] <= rect2['x'],
                rect2['x'] + rect2['width'] <= rect1['x'],
                rect1['y'] + rect1['height'] <= rect2['y'],
                rect2['y'] + rect2['height'] <= rect1['y']
            ])

# Solve the model
solver = cp_model.CpSolver()
start_time = time.time()
status = solver.Solve(model)
end_time = time.time()

# Output solution
if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
    print(f"Solution found in {end_time - start_time:.2f} seconds.")
    solution_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    
    for label, (coords, rect) in enumerate(rectangles.items(), start=1):
        x = solver.Value(rect['x'])
        y = solver.Value(rect['y'])
        width = solver.Value(rect['width'])
        height = solver.Value(rect['height'])
        for i in range(width):
            for j in range(height):
                solution_grid[y + j][x + i] = label
    
    # Print the solution grid
    for row in solution_grid:
        print(" ".join(str(cell) if cell > 0 else "." for cell in row))
else:
    print("No solution exists.")