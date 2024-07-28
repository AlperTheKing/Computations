import sympy
from mpmath import mp
import multiprocessing as multip
import os
import time

# Calculate the first 100,000 decimal places of e
mp.dps = 100001  # Set the number of decimal places
e_str = str(mp.e)[2:]  # Remove the '2.' part

# Print the first 1000 decimal places of e
print(f"The first 1000 digits of e: {e_str[:1000]}")

# Function to check for primes in a given range
def find_primes(start, end):
    local_primes = []
    for i in range(start, end):
        num = e_str[i:i + 10]
        if len(num) == 10 and sympy.isprime(int(num)):
            local_primes.append((i, num))
    return local_primes

# Main function to handle multiprocessing
def main():
    start_time = time.time()  # Start time measurement

    # Multiprocessing setup
    num_processes = os.cpu_count()
    step = len(e_str) // num_processes

    # Create a pool of processes
    with multip.Pool(processes=num_processes) as pool:
        # Distribute the task across the processes
        results = pool.starmap(find_primes, [(i * step, min((i + 1) * step, len(e_str) - 9)) for i in range(num_processes)])

    # Combine results from all processes
    prime_indices = [item for sublist in results for item in sublist]

    # Print the found primes and their indices
    for index, prime in prime_indices:
        print(f"Index: {index}, Prime: {prime}")

    end_time = time.time()  # End time measurement
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
