import numpy as np
import multiprocessing
import os

# Number of simulations per process
num_simulations_per_process = 1000  # Adjust this as needed

# Total number of simulations
total_simulations = 10000  # 10,000 simulations

# Target number of steps
target_steps = 10_000

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

# Calculate the number of processes needed
num_processes = total_simulations // num_simulations_per_process

# Ensure the number of processes does not exceed the number of CPU cores available
num_processes = min(num_processes, os.cpu_count())

# Use multiprocessing.Pool for multi-processing
if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Divide the work among the processes
        num_simulations_list = [num_simulations_per_process] * num_processes

        # Collect the results
        results = pool.map(run_simulation, num_simulations_list)

    # Calculate the total successful simulations
    total_successful_simulations = sum(results)

    # Calculate the probability
    probability = total_successful_simulations / total_simulations

    print(f"Probability of reaching exactly {target_steps} steps: {probability:.10f}")
