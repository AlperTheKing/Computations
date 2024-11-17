import math

def sieve_of_eratosthenes(limit):
    """
    Generates all prime numbers up to the specified limit using the Sieve of Eratosthenes.
    """
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]  # 0 and 1 are not primes
    for num in range(2, int(math.isqrt(limit)) + 1):
        if sieve[num]:
            for multiple in range(num*num, limit + 1, num):
                sieve[multiple] = False
    primes = [num for num, is_prime in enumerate(sieve) if is_prime]
    return primes

def legendre_exponent(n, p):
    """
    Computes the exponent of prime p in the factorization of n! using Legendre's formula.
    """
    exponent = 0
    power = p
    while power <= n:
        exponent += n // power
        power *= p
    return exponent

def compute_total_exponents(max_n, primes):
    """
    Computes the total exponents of each prime in the product P = 1! * 2! * ... * max_n!.
    """
    total_exponents = {p: 0 for p in primes}
    for p in primes:
        for k in range(1, max_n + 1):
            total_exponents[p] += legendre_exponent(k, p)
    return total_exponents

def find_smallest_n(max_n, primes, total_exponents):
    """
    Finds the smallest n such that dividing P by n! results in all even exponents.
    """
    # Precompute the exponents of n! for all n up to max_n
    # Initialize a dictionary to keep track of exponents in n!
    factorial_exponents = {p: 0 for p in primes}
    
    for n in range(1, max_n + 1):
        temp = n
        for p in primes:
            if p > temp:
                break
            while temp % p == 0:
                factorial_exponents[p] += 1
                temp //= p
        # Check if (total_exponents[p] - factorial_exponents[p]) is even for all primes
        is_perfect_square = True
        for p in primes:
            if (total_exponents[p] - factorial_exponents[p]) % 2 != 0:
                is_perfect_square = False
                break
        if is_perfect_square:
            return n  # Found the smallest n
    return None  # If no such n is found within the range

def sum_of_digits(number):
    """
    Calculates the sum of the digits of the given number.
    """
    return sum(int(digit) for digit in str(number))

def main():
    max_n = 2016
    # Step 1: Find all prime numbers up to max_n
    primes = sieve_of_eratosthenes(max_n)
    
    # Step 2: Compute total exponents of each prime in P
    total_exponents = compute_total_exponents(max_n, primes)
    
    # Step 3: Find the smallest n such that P / n! is a perfect square
    smallest_n = find_smallest_n(max_n, primes, total_exponents)
    
    if smallest_n is not None:
        # Step 4: Calculate the sum of the digits of n
        digit_sum = sum_of_digits(smallest_n)
        print(f"The smallest integer n is: {smallest_n}")
        print(f"The sum of the digits of n is: {digit_sum}")
    else:
        print("No such integer n found within the given range.")

if __name__ == "__main__":
    main()
