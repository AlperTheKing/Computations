import numpy as np

# Number of dice rolls
num_rolls = 1_000_000_000

# Simulate dice rolls for three dice
rolls1 = np.random.randint(1, 7, size=num_rolls)
rolls2 = np.random.randint(1, 7, size=num_rolls)
rolls3 = np.random.randint(1, 7, size=num_rolls)

# Calculate the sum of all rolls for the three dice
total_sum = np.sum(rolls1 + rolls2 + rolls3)

# Calculate the average result
average_result = total_sum / num_rolls

print(average_result)