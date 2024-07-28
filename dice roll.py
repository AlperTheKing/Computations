import numpy as np

# Number of dice rolls
num_rolls = 1_000_000_000

# Simulate dice rolls
rolls = np.random.randint(1, 7, size=num_rolls)

# Calculate the sum of all rolls
total_sum = np.sum(rolls)

# Calculate the average result
average_result = total_sum / num_rolls

print(average_result)