import numpy as np
import concurrent.futures

# Number of simulations per thread
num_simulations_per_thread = 1000  # Adjust this as needed

# Total number of simulations
total_simulations = 10000  # Adjust this as needed

# Target number of steps
target_steps = 1_000_000

# Function to run the simulation
def run_simulation(num_simulations):
    successful_simulations = 0
    for _ in range(num_simulations):
        current_position = 0
        while current_position < target_steps:
            toss = np.random.randint(0, 2)
            if toss == 0:
                current_position += 1
            else:
                current_position += 2

            if current_position > target_steps:
                break

        if current_position == target_steps:
            successful_simulations += 1
    return successful_simulations

# Calculate the number of threads needed
num_threads = total_simulations // num_simulations_per_thread

# Use ThreadPoolExecutor for multi-threading
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(run_simulation, num_simulations_per_thread) for _ in range(num_threads)]

    # Collect the results
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

# Calculate the total successful simulations
total_successful_simulations = sum(results)

# Calculate the probability
probability = total_successful_simulations / total_simulations

print(f"Probability of reaching exactly {target_steps} steps: {probability:.10f}")