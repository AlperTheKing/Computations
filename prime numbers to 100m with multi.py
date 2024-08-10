import multiprocessing as multi
import os
import time  # Importing the time module

# Optimized prime number checking function using modular arithmetic
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    if n % 5 == 0:
        return n == 5
    if n % 7 == 0:
        return n == 7
    if n % 11 == 0:
        return n == 11
    if n % 13 == 0:
        return n == 13
    if n % 17 == 0:
        return n == 17
    if n % 19 == 0:
        return n == 19
    i = 23
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0 or n % (i + 6) == 0:
            return False
        i += 6
    return True

# Function to find prime numbers in a given range
def find_primes_in_range(start, end, prime_list, idx):
    try:
        primes = []
        for num in range(start, end):
            if is_prime(num):
                primes.append(num)
        prime_list[idx] = primes
        print(f"Processed range {start}-{end}, found {len(primes)} primes")
    except Exception as e:
        print(f"Error processing range {start}-{end}: {e}")

# Main function
def main():
    start_time = time.time()  # Start timing
    max_num = 100_000_000
    num_processes = min(60, os.cpu_count())  # Dynamically determine number of processes based on CPU cores
    chunk_size = max_num // num_processes

    with multi.Manager() as manager:
        prime_list = manager.dict()  # Use a dictionary to store primes by index
        processes = []

        # Create ranges and processes
        for i in range(num_processes):
            start = i * chunk_size
            end = start + chunk_size if i < num_processes - 1 else max_num
            process = multi.Process(target=find_primes_in_range, args=(start, end, prime_list, i))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
            print(f"Joined process {process.pid}")

        print("All processes joined.")
        
        # Collect and flatten the list of primes
        primes = []
        for i in range(num_processes):
            if i in prime_list:
                primes.extend(prime_list[i])
            else:
                print(f"Warning: Process {i} did not return any results.")
        
        primes.sort()

        # Debugging output to check if primes are collected
        if primes:
            print(f"Collected {len(primes)} primes successfully.")
        else:
            print("No primes collected. Something went wrong.")

    end_time = time.time()  # End timing
    print(f"Total execution time: {end_time - start_time:.2f} seconds")  # Print the execution time

    return primes

if __name__ == "__main__":
    primes = main()
    if primes:
        print(f"Found {len(primes)} prime numbers.")
        print(f"First 10 primes: {primes[:10]}")
        print(f"Last 10 primes: {primes[-10:]}")
    else:
        print("No primes found or an issue occurred with collecting results.")
