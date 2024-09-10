import itertools

# Define the characters
letters = ['A', 'A', 'B', 'C', 'D', 'E']

# Function to check if A's are adjacent
def has_adjacent_A(code):
    for i in range(len(code) - 1):
        if code[i] == 'A' and code[i + 1] == 'A':
            return True
    return False

# Generate all unique permutations
all_permutations = set(itertools.permutations(letters))

# Filter out the ones where A's are adjacent
valid_codes = [perm for perm in all_permutations if not has_adjacent_A(perm)]

# Total number of valid codes
num_valid_codes = len(valid_codes)
print(num_valid_codes)