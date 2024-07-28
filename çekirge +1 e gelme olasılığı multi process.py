import numpy as np
import multiprocessing
import time

# Number of simulations
num_simulations = 100_0000

# Number of steps
num_steps = 100_000

# Number of processes (cores) to use
num_processes = 16

# Number of simulations per process
simulations_per_process = num_simulations // num_processes

# Function to run multiple simulations in a single process
def run_simulations(num_simulations):
    successful_simulations = 0
    for _ in range(num_simulations):
        position = 0
        for _ in range(num_steps):
            toss = np.random.randint(0, 2)
            if toss == 0:
                position += 1
            else:
                position -= 1

            # Check if the grasshopper passed +1 position
            if position == 1:
                successful_simulations += 1
                break
    return successful_simulations

# Main function to handle multiprocessing and measure time
def main():
    start_time = time.time()

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the simulations across the processes
        results = pool.map(run_simulations, [simulations_per_process] * num_processes)

    # Calculate the total successful simulations
    total_successful_simulations = sum(results)

    # Calculate the probability
    probability = total_successful_simulations / num_simulations

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Probability of passing +1 position: {probability:.10f}")
    print(f"Time taken to complete: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()