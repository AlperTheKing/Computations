import math

# Function to compute LCM of three numbers
def lcm(x, y, z):
    return math.lcm(math.lcm(x, y), z)

# Function to find all triples (a, b, c) such that LCM(a, b, c) = a + b + c
def find_triples(max_value):
    solutions = []
    for a in range(1, max_value + 1):
        for b in range(a, max_value + 1):  # Start b from a to avoid duplicate pairs
            for c in range(b, max_value + 1):  # Start c from b to avoid duplicate pairs
                if lcm(a, b, c) == a + b + c:
                    solutions.append((a, b, c))
    return solutions

# Define the maximum value to explore
max_value = 100  # Adjust this range as needed

# Find solutions
solutions = find_triples(max_value)

# Print the solutions
if solutions:
    for solution in solutions:
        print(f"a={solution[0]}, b={solution[1]}, c={solution[2]}")
else:
    print("No solutions found")
