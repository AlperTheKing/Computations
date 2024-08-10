import multiprocessing as multi

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
def find_primes_in_range(start, end, queue):
    primes = []
    for num in range(start, end):
        if is_prime(num):
            primes.append(num)
    queue.put(primes)
    print(f"Processed range {start}-{end}, found {len(primes)} primes")

# Main function
def main():
    max_num = 100_000_000
    num_processes = 60
    chunk_size = max_num // num_processes

    queue = multi.Queue()
    processes = []

    # Create ranges and processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else max_num
        process = multi.Process(target=find_primes_in_range, args=(start, end, queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        print(f"Joined process {process.pid}")

    print("All processes joined.")
    
    primes = []
    while not queue.empty():
        primes.extend(queue.get())
    
    return primes

if __name__ == "__main__":
    primes = main()
    if primes:
        print(f"Found {len(primes)} prime numbers.")
    else:
        print("No primes found or an issue occurred with collecting results.")
