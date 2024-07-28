import numpy as np
import time

# Number of simulations
num_simulations = 100_000

# Number of steps
num_steps = 100_000

# Function to run a single simulation
def run_simulation():
    position = 0
    for _ in range(num_steps):
        toss = np.random.randint(0, 2)
        if toss == 0:
            position += 1
        else:
            position -= 1

        # Check if the grasshopper passed +1 position
        if position == 1:
            return 1
    return 0

# Main function to run all simulations and measure time
def main():
    start_time = time.time()

    successful_simulations = 0
    for _ in range(num_simulations):
        successful_simulations += run_simulation()

    # Calculate the probability
    probability = successful_simulations / num_simulations

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Probability of passing +1 position: {probability:.10f}")
    print(f"Time taken to complete: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()