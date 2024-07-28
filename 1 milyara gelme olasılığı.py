import numpy as np

# Number of simulations
num_simulations = 1000  # Using a smaller number for illustration

# Target number of steps
target_steps = 1_000_0

# Initialize a counter for successful simulations
successful_simulations = 0

# Perform the simulations
for _ in range(num_simulations):
    current_position = 0
    while current_position < target_steps:
        # Simulate a coin toss: 0 for heads (1 step), 1 for tails (2 steps)
        toss = np.random.randint(0, 2)
        if toss == 0:
            current_position += 1
        else:
            current_position += 2

        # If we exceed the target in a single step, we can't reach exactly the target
        if current_position > target_steps:
            break

    if current_position == target_steps:
        successful_simulations += 1

# Calculate the probability
probability = successful_simulations / num_simulations

print(f"Probability of reaching exactly {target_steps} steps: {probability:.10f}")