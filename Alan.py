from ortools.sat.python import cp_model
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the grid dimensions and initial grid with numbers
GRID_SIZE = 9
initial_grid = [
    [0, 0, 8, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 3, 0, 0, 3, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 9, 0, 10, 0, 4, 0, 6],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 3, 0, 0, 5, 0, 0, 0, 0],
    [4, 0, 0, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# Define the areas for each rectangle based on the initial grid numbers
area_constraints = {
    (0, 2): 8,
    (1, 0): 2,
    (1, 3): 3,
    (1, 6): 3,
    (2, 3): 3,
    (4, 0): 3,
    (4, 2): 9,
    (4, 4): 10,
    (4, 6): 4,
    (4, 8): 6,
    (5, 7): 8,
    (6, 1): 3,
    (6, 4): 5,
    (7, 0): 4,
    (7, 3): 10
}

# Define the model
model = cp_model.CpModel()

# Create variables for the coordinates of each rectangle with unique IDs
rectangles = {}
for idx, ((x, y), area) in enumerate(area_constraints.items(), start=1):
    rectangles[idx] = {
        'init_pos': (x, y),
        'area': area,
        'x': model.NewIntVar(0, GRID_SIZE - 1, f"x_{idx}"),
        'y': model.NewIntVar(0, GRID_SIZE - 1, f"y_{idx}")
    }

# Function to generate all possible (width, height) pairs for a given area
def generate_width_height_pairs(area):
    pairs = []
    for w in range(1, min(area, GRID_SIZE) + 1):
        if area % w == 0:
            h = area // w
            if h <= GRID_SIZE:
                pairs.append((w, h))
    return pairs

# Create width and height variables with enumerated possible pairs
for idx, rect in rectangles.items():
    area = rect['area']
    init_x, init_y = rect['init_pos']
    pairs = generate_width_height_pairs(area)
    
    if not pairs:
        raise ValueError(f"No valid (width, height) pairs for rectangle {idx} with area {area}")
    
    print(f"Rectangle {idx}: Area={area}, Initial Position={init_x, init_y}, Possible Pairs={pairs}")
    
    # Create a boolean variable for each possible pair
    pair_vars = []
    for p_idx, (w, h) in enumerate(pairs):
        pair_var = model.NewBoolVar(f"rect{idx}_pair{p_idx+1}")
        pair_vars.append(pair_var)
    
    # Ensure exactly one pair is chosen
    model.AddExactlyOne(pair_vars)
    
    # Define width and height variables as integer variables
    rect['width'] = model.NewIntVar(1, GRID_SIZE, f"width_{idx}")
    rect['height'] = model.NewIntVar(1, GRID_SIZE, f"height_{idx}")
    
    # Link width and height to the chosen pair
    for p_idx, (w, h) in enumerate(pairs):
        model.Add(rect['width'] == w).OnlyEnforceIf(pair_vars[p_idx])
        model.Add(rect['height'] == h).OnlyEnforceIf(pair_vars[p_idx])
    
    # Keep rectangles within grid bounds
    model.Add(rect['x'] + rect['width'] <= GRID_SIZE)
    model.Add(rect['y'] + rect['height'] <= GRID_SIZE)
    
    # Ensure the rectangle includes the designated initial cell
    model.Add(rect['x'] <= init_x)
    model.Add(init_x < rect['x'] + rect['width'])
    model.Add(rect['y'] <= init_y)
    model.Add(init_y < rect['y'] + rect['height'])

# Ensure rectangles do not overlap
rectangle_ids = list(rectangles.keys())
for i in range(len(rectangle_ids)):
    for j in range(i + 1, len(rectangle_ids)):
        rect1 = rectangles[rectangle_ids[i]]
        rect2 = rectangles[rectangle_ids[j]]
        
        # Create Boolean variables for each non-overlapping condition
        b1 = model.NewBoolVar(f"rect{rectangle_ids[i]}_left_of_rect{rectangle_ids[j]}")
        b2 = model.NewBoolVar(f"rect{rectangle_ids[i]}_right_of_rect{rectangle_ids[j]}")
        b3 = model.NewBoolVar(f"rect{rectangle_ids[i]}_above_rect{rectangle_ids[j]}")
        b4 = model.NewBoolVar(f"rect{rectangle_ids[i]}_below_rect{rectangle_ids[j]}")
        
        # Link the boolean variables with the actual conditions
        model.Add(rect1['x'] + rect1['width'] <= rect2['x']).OnlyEnforceIf(b1)
        model.Add(rect2['x'] + rect2['width'] <= rect1['x']).OnlyEnforceIf(b2)
        model.Add(rect1['y'] + rect1['height'] <= rect2['y']).OnlyEnforceIf(b3)
        model.Add(rect2['y'] + rect2['height'] <= rect1['y']).OnlyEnforceIf(b4)
        
        # Add the disjunction: at least one of these must be true
        model.AddBoolOr([b1, b2, b3, b4])

# Symmetry-breaking constraints (Removed to allow full flexibility)
# for idx in range(2, len(rectangle_ids)+1):
#     model.Add(rectangles[idx]['x'] >= rectangles[idx-1]['x'])
#     model.Add(rectangles[idx]['y'] >= rectangles[idx-1]['y'])

# Solve the model
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 120.0  # Increased time limit for debugging
start_time = time.time()
status = solver.Solve(model)
end_time = time.time()

# Function to visualize the solution using matplotlib
def visualize_solution(solution_grid, rectangles, GRID_SIZE):
    fig, ax = plt.subplots(figsize=(6,6))
    # Draw grid lines
    for x in range(GRID_SIZE + 1):
        ax.axhline(x, lw=1, color='k', zorder=5)
        ax.axvline(x, lw=1, color='k', zorder=5)
    
    # Define a color map
    cmap = plt.get_cmap('tab20')
    
    # Assign unique colors to each rectangle
    colors = {idx: cmap(idx % 20) for idx in rectangles.keys()}
    
    # Draw rectangles
    for idx, rect in rectangles.items():
        x = solver.Value(rect['x'])
        y = solver.Value(rect['y'])
        width = solver.Value(rect['width'])
        height = solver.Value(rect['height'])
        rect_patch = patches.Rectangle((x, GRID_SIZE - y - height), width, height, linewidth=1, edgecolor='black', facecolor=colors[idx], alpha=0.5)
        ax.add_patch(rect_patch)
        # Annotate rectangle with its ID and area
        ax.text(x + width/2, GRID_SIZE - y - height + height/2, f"R{idx}\n{rect['area']}", 
                horizontalalignment='center', verticalalignment='center', fontsize=8, color='black')
    
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

# Output solution
if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
    print(f"Solution found in {end_time - start_time:.2f} seconds.")
    solution_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    
    for idx, rect in rectangles.items():
        area = rect['area']
        x = solver.Value(rect['x'])
        y = solver.Value(rect['y'])
        width = solver.Value(rect['width'])
        height = solver.Value(rect['height'])
        for i in range(width):
            for j in range(height):
                cell_x = x + i
                cell_y = y + j
                if solution_grid[cell_y][cell_x] != 0:
                    print(f"Overlap detected at ({cell_x}, {cell_y}) between rectangles {solution_grid[cell_y][cell_x]} and {idx}")
                # Label cells with rectangle IDs
                solution_grid[cell_y][cell_x] = idx
    
    # Print the solution grid with unique labels
    print("\nSolution Grid (R=Rectangle ID):")
    for row in solution_grid:
        print(" ".join(f"R{cell}" if cell > 0 else "." for cell in row))
    
    # Visualize the solution
    visualize_solution(solution_grid, rectangles, GRID_SIZE)
else:
    print("No solution exists.")