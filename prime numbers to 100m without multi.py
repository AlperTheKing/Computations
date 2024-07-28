import time

def find_prime_numbers(limit):
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0], is_prime[1] = False, False

    for num in range(2, limit + 1):
        if is_prime[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                is_prime[multiple] = False

    return primes

limit = 100_000_000

start_time = time.time()  # Start time measurement
prime_numbers = find_prime_numbers(limit)
end_time = time.time()  # End time measurement

# Print the number of primes found and the last 10 prime numbers
print(f"Total prime numbers up to {limit}: {len(prime_numbers)}")
print("Last 10 prime numbers:", prime_numbers[-10:])
print(f"Calculation took {end_time - start_time:.2f} seconds")
