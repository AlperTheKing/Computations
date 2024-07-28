import multiprocessing as mp
import time

# Divisibility rule functions
def is_divisible_by_2(n):
    return n % 2 == 0

def is_divisible_by_3(n):
    return sum(int(digit) for digit in str(n)) % 3 == 0

def is_divisible_by_5(n):
    return n % 10 == 0 or n % 10 == 5

def is_divisible_by_7(n):
    return n % 7 == 0

def is_divisible_by_11(n):
    return n % 11 == 0

def is_divisible_by_13(n):
    return n % 13 == 0

def is_divisible_by_17(n):
    return n % 17 == 0

def is_divisible_by_19(n):
    return n % 19 == 0

def is_divisible_by_23(n):
    return n % 23 == 0

def is_divisible_by_29(n):
    return n % 29 == 0

def is_divisible_by_31(n):
    return n % 31 == 0

def is_divisible_by_37(n):
    return n % 37 == 0

def is_divisible_by_41(n):
    return n % 41 == 0

def is_divisible_by_43(n):
    return n % 43 == 0

def is_divisible_by_47(n):
    return n % 47 == 0

def is_divisible_by_53(n):
    return n % 53 == 0

def is_divisible_by_59(n):
    return n % 59 == 0

def is_divisible_by_61(n):
    return n % 61 == 0

def is_divisible_by_67(n):
    return n % 67 == 0

def is_divisible_by_71(n):
    return n % 71 == 0

def is_divisible_by_73(n):
    return n % 73 == 0

def is_divisible_by_79(n):
    return n % 79 == 0

def is_divisible_by_83(n):
    return n % 83 == 0

def is_divisible_by_89(n):
    return n % 89 == 0

def is_divisible_by_97(n):
    return n % 97 == 0

# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    if is_divisible_by_2(n) and n != 2:
        return False
    if is_divisible_by_3(n) and n != 3:
        return False
    if is_divisible_by_5(n) and n != 5:
        return False
    if is_divisible_by_7(n) and n != 7:
        return False
    if is_divisible_by_11(n) and n != 11:
        return False
    if is_divisible_by_13(n) and n != 13:
        return False
    if is_divisible_by_17(n) and n != 17:
        return False
    if is_divisible_by_19(n) and n != 19:
        return False
    if is_divisible_by_23(n) and n != 23:
        return False
    if is_divisible_by_29(n) and n != 29:
        return False
    if is_divisible_by_31(n) and n != 31:
        return False
    if is_divisible_by_37(n) and n != 37:
        return False
    if is_divisible_by_41(n) and n != 41:
        return False
    if is_divisible_by_43(n) and n != 43:
        return False
    if is_divisible_by_47(n) and n != 47:
        return False
    if is_divisible_by_53(n) and n != 53:
        return False
    if is_divisible_by_59(n) and n != 59:
        return False
    if is_divisible_by_61(n) and n != 61:
        return False
    if is_divisible_by_67(n) and n != 67:
        return False
    if is_divisible_by_71(n) and n != 71:
        return False
    if is_divisible_by_73(n) and n != 73:
        return False
    if is_divisible_by_79(n) and n != 79:
        return False
    if is_divisible_by_83(n) and n != 83:
        return False
    if is_divisible_by_89(n) and n != 89:
        return False
    if is_divisible_by_97(n) and n != 97:
        return False
    # General prime check for numbers greater than 97
    for i in range(101, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Function to find prime numbers in a specific range
def find_primes_in_range(start, end, queue):
    primes = [num for num in range(start, end) if is_prime(num)]
    queue.put(primes)
    print(f"Processed range {start}-{end}, found {len(primes)} primes")

# Function to find prime numbers up to a specified limit using multiprocessing
def find_prime_numbers(limit):
    num_processes = mp.cpu_count()  # Number of CPU cores
    chunk_size = limit // num_processes  # Size of each chunk

    queue = mp.Queue()
    processes = []

    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else limit
        process = mp.Process(target=find_primes_in_range, args=(start, end, queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    primes = []
    while not queue.empty():
        primes.extend(queue.get())

    return primes

if __name__ == "__main__":
    limit = 1_000_0000  # Adjusted limit for demonstration
    start_time = time.time()
    prime_numbers = find_prime_numbers(limit)
    end_time = time.time()

    # Print the total number of prime numbers and the last 10 prime numbers
    print(f"Total prime numbers up to {limit}: {len(prime_numbers)}")
    print("Last 10 prime numbers:", prime_numbers[-10:])
    print(f"Calculation took {end_time - start_time:.2f} seconds")
