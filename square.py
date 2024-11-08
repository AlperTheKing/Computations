import math
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np

# Define the grid size (18x18)
GRID_SIZE = 18

# Define the original shapes with (0, 0) as the origin
# Each shape is a list of (x, y) coordinates relative to its origin
original_shapes = {
    1: [(0, 0), (6, 0), (12, 0), (0, 6), (12, 6)],               # Shape 1
    2: [(0, 0), (6, 6), (0, 6), (0, 12), (3, 15), (6, 12)],    # Shape 2
    3: [(0, 0), (12, 6), (6, 6), (6, 12)],                      # Shape 3
    4: [(0, 0), (6, 12), (3, 15), (6, 18), (9, 15)],            # Shape 4
    5: [(0, 0), (12, 6), (12, 18), (6, 18), (9, 15), (6, 12)]  # Shape 5
}

def rotate_shape(shape_coords, angle):
    """
    Rotate shape coordinates by the given angle.
    Angle should be one of [0, 45, 90, 135, 180, 225, 270, 315].
    Returns rotated coordinates rounded to nearest integer.
    """
    radians = math.radians(angle)
    rotated = []
    for x, y in shape_coords:
        rotated_x = x * math.cos(radians) - y * math.sin(radians)
        rotated_y = x * math.sin(radians) + y * math.cos(radians)
        # Round to nearest integer to align with grid
        rotated.append((round(rotated_x), round(rotated_y)))
    # Normalize coordinates to have (0,0) as the minimum x and y
    min_x = min(x for x, y in rotated)
    min_y = min(y for x, y in rotated)
    normalized = [(x - min_x, y - min_y) for x, y in rotated]
    return normalized

def generate_rotated_shapes(original_shapes):
    """
    Generate all rotated variants for each original shape at 45-degree increments.
    Returns a dictionary with keys as (shape_id, rotation_angle) and values as coordinates.
    """
    rotated_shapes = {}
    for shape_id, coords in original_shapes.items():
        for angle in range(0, 360, 45):
            rotated = rotate_shape(coords, angle)
            rotated_shapes[(shape_id, angle)] = rotated
    return rotated_shapes

# Generate all rotated shapes
rotated_shapes = generate_rotated_shapes(original_shapes)

# Assign unique IDs to each rotated variant
shape_variants = {}
for (shape_id, angle), coords in rotated_shapes.items():
    variant_id = f'Shape{shape_id}_{angle}'
    shape_variants[variant_id] = coords

# Initialize the CP-SAT model
model = cp_model.CpModel()

# Function to check if a shape placement is valid within the grid
def is_valid_placement(shape_coords, origin_i, origin_j, grid_size):
    for dx, dy in shape_coords:
        x = origin_i + dx
        y = origin_j + dy
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False
    return True

# Precompute feasible placements and create variables
placements = {}
for variant_id, coords in shape_variants.items():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if is_valid_placement(coords, i, j, GRID_SIZE):
                var = model.NewBoolVar(f'place_{variant_id}_at_{i}_{j}')
                placements[(variant_id, i, j)] = var

# Create a mapping from cells to placements that cover them
cell_to_placements = {}
for (variant_id, i, j), var in placements.items():
    for dx, dy in shape_variants[variant_id]:
        x = i + dx
        y = j + dy
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            cell_to_placements.setdefault((x, y), []).append(var)

# Define variables for the bounding square
# Fix the bounding square to start at (0,0) to simplify constraints
square_min_i = 0
square_min_j = 0
# Define square_size as a variable to minimize
square_size = model.NewIntVar(1, GRID_SIZE, 'square_size')

# Calculate total area covered by all shapes
total_shape_area = sum(len(coords) for coords in original_shapes.values())

# Ensure that square_size is at least ceil(sqrt(total_shape_area))
min_square_size = math.ceil(math.sqrt(total_shape_area))
model.Add(square_size >= min_square_size)

# Ensure that all shapes lie within the bounding square
for (variant_id, i, j), var in placements.items():
    coords = shape_variants[variant_id]
    for dx, dy in coords:
        x = i + dx
        y = j + dy
        # x < square_size
        model.Add(x < square_size).OnlyEnforceIf(var)
        # y < square_size
        model.Add(y < square_size).OnlyEnforceIf(var)

# Ensure the bounding square is within grid boundaries
model.Add(square_size <= GRID_SIZE)

# Enforce that each shape is placed exactly once
for shape_id in original_shapes.keys():
    # Get all variants of this shape
    variants = [variant_id for variant_id in shape_variants if variant_id.startswith(f'Shape{shape_id}_')]
    # Sum of all placements for all variants of this shape must be exactly 1
    model.Add(
        sum([
            placements[(variant_id, i, j)]
            for variant_id in variants
            for (v_id, i, j) in placements
            if v_id == variant_id
        ]) == 1
    )

# Prevent overlaps: each cell can be covered by at most one shape
for (i, j), vars_covering in cell_to_placements.items():
    model.Add(sum(vars_covering) <= 1)

# Define the objective: minimize the size of the bounding square
model.Minimize(square_size)

# Solve the model
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 300.0  # Set a time limit of 5 minutes
solver.parameters.log_search_progress = True    # Enable logs for debugging
status = solver.Solve(model)

# Function to visualize the solution using matplotlib
def visualize_solution(grid, square_min_i_val, square_min_j_val, square_size_val):
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('tab20', len(shape_variants) + 1)  # +1 for empty cells
    plt.imshow(grid, cmap=cmap, origin='upper')

    # Draw the bounding square
    square = plt.Rectangle((square_min_j_val, square_min_i_val), square_size_val, square_size_val,
                           linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(square)

    # Annotate each shape
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] != 0:
                plt.text(j, i, grid[i][j], ha='center', va='center', color='white', fontsize=6)

    # Create a colorbar with shape labels
    cbar = plt.colorbar(ticks=range(len(shape_variants)+1))
    labels = ['Empty'] + list(shape_variants.keys())
    cbar.ax.set_yticklabels(labels[:len(shape_variants)+1])
    plt.title(f'Tiling Solution (Square Size: {square_size_val}x{square_size_val})')
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.show()

# Output solution if found
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution found:")
    # Initialize an empty grid with 0 representing empty cells
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Assign a unique number to each shape variant for visualization
    variant_to_number = {variant: idx+1 for idx, variant in enumerate(shape_variants.keys())}

    # Iterate through all placements and mark the grid
    for (variant_id, i, j), var in placements.items():
        if solver.Value(var) == 1:
            shape_number = variant_to_number[variant_id]
            for dx, dy in shape_variants[variant_id]:
                x = i + dx
                y = j + dy
                grid[x][y] = shape_number  # Assign the shape's unique number

    # Retrieve square bounding box values
    square_min_i_val = square_min_i
    square_min_j_val = square_min_j
    square_size_val = solver.Value(square_size)

    # Print the grid
    for row in grid:
        print(' '.join(['.' if cell == 0 else str(cell) for cell in row]))

    # Visualize the solution
    visualize_solution(grid, square_min_i_val, square_min_j_val, square_size_val)
else:
    print("No solution found.")
