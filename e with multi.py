import sympy
from mpmath import mp
import multiprocessing as multi
import time

# Set the number of decimal places for mpmath
mp.dps = 10000001  # Set precision to 10 million decimal places
e_str = str(mp.e)[2:]  # Remove the '2.' part from the string representation

print(f"Calculated the first 10 million digits of e.")

# Function to check for primes in a given range
def find_primes(start, end, return_list):
    local_primes = []
    for i in range(start, end):
        num = e_str[i:i + 10]
        if len(num) == 10 and num[0] != '0' and sympy.isprime(int(num)):
            local_primes.append((i, num))
    return_list.extend(local_primes)

# Measure the start time
start_time = time.time()

# Multi-processing setup
num_processes = 48
step = len(e_str) // num_processes
manager = multi.Manager()
return_list = manager.list()
processes = []

# Create and start processes
for i in range(num_processes):
    start = i * step
    end = min(start + step, len(e_str) - 9)
    process = multi.Process(target=find_primes, args=(start, end, return_list))
    processes.append(process)
    process.start()

# Wait for all processes to complete
for process in processes:
    process.join()

# Measure the end time
end_time = time.time()

# Print the found primes and their indices
for index, prime in return_list:
    print(f"Index: {index}, Prime: {prime}")

# Print the total execution time
print(f"Total execution time: {end_time - start_time:.2f} seconds")
