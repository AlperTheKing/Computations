import itertools
import math

# Define points with their coordinates
points = {
    'A': (0, 0), 'B': (0, 5), 'C': (2, 2),
    'D': (3, 0), 'E': (4, 4), 'F': (5, 1),
    'G': (5, 6), 'H': (6, 0), 'J': (6, 2)
}

# Function to calculate Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((points[p1][0] - points[p2][0]) ** 2 + (points[p1][1] - points[p2][1]) ** 2)

# Generate all possible paths
all_paths = list(itertools.permutations(points.keys()))

# Function to check if a path has strictly increasing distances
def is_valid_path(path):
    distances = []
    for i in range(len(path) - 1):
        d = distance(path[i], path[i + 1])
        if distances and d <= distances[-1]:  # Break if distances are not strictly increasing
            return False
        distances.append(d)
    return True

# Find and print the valid path(s)
valid_paths = []
for index, path in enumerate(all_paths):
    if is_valid_path(path):
        valid_paths.append(path)
    if index % 10000 == 0:  # Progress indicator
        print(f"Checked {index} paths...")

# Output all valid paths or indicate if none found
print(valid_paths if valid_paths else "No valid path found")