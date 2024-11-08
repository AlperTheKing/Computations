def sieve_smallest_prime_factors(limit):
    """
    Sieve of Eratosthenes to compute the smallest prime factor for every number up to limit.
    Returns a list spf where spf[n] is the smallest prime factor of n.
    """
    spf = [0] * (limit + 1)
    spf[0], spf[1] = 0, 1  # 0 and 1 are not prime
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i  # i is prime
            for multiple in range(i * i, limit + 1, i):
                if spf[multiple] == 0:
                    spf[multiple] = i
    return spf

def count_valid_numbers(limit):
    """
    Counts the number of natural numbers less than 'limit' where:
    1. The smallest prime divisor is p.
    2. p^2 + p + 1 divides the number.
    """
    spf = sieve_smallest_prime_factors(limit)
    count = 0

    for n in range(2, limit):
        p = spf[n]  # Smallest prime factor of n
        condition_divisor = p**2 + p + 1
        if condition_divisor > n:
            continue  # p^2 + p + 1 cannot divide n if it's greater than n
        if n % condition_divisor == 0:
            count += 1

    return count

# Define the upper limit
limit = 2013

# Calculate the result
result = count_valid_numbers(limit)

print(f"Number of natural numbers less than {limit} that satisfy the conditions: {result}")
