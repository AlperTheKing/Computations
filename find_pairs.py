import math

# Function to find all pairs (x, y, z) such that 3^x - 5^y = z^2
def find_solutions(max_x, max_y):
    solutions = []
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            value = 3**x - 5**y
            if value > 0:
                z = math.isqrt(value)  # Compute the integer square root of value
                if z * z == value:  # Check if value is a perfect square
                    solutions.append((x, y, z))
    return solutions

# Define the range to explore
max_x = 1000  # Adjust this range as needed
max_y = 1000  # Adjust this range as needed

# Find solutions
solutions = find_solutions(max_x, max_y)

# Print the solutions
if solutions:
    for solution in solutions:
        print(f"x={solution[0]}, y={solution[1]}, z={solution[2]}")
else:
    print("No solutions found")