def split_digits(n):
    return [int(digit) for digit in str(n)]

def divisible_by_2(n):
    digits = split_digits(n)
    return digits[-1] in [0, 2, 4, 6, 8]

def divisible_by_3(n):
    digits = split_digits(n)
    return sum(digits) % 3 == 0

def divisible_by_5(n):
    digits = split_digits(n)
    return digits[-1] == 5 or digits[-1] == 0

def divisible_by_7(n):
    digits = split_digits(n)
    if len(digits) == 1:
        return False
    combined_number = int(''.join(map(str, digits[:-1])))
    last_digit = digits[-1]
    while combined_number >= 10:
        combined_number -= last_digit * 2
    return combined_number % 7 == 0

def find_m(D):
    last_digit = split_digits(D)
    if last_digit[-1] == 1:
        return 9
    elif last_digit[-1] == 3:
        return 3
    elif last_digit[-1] == 7:
        return 7
    elif last_digit[-1] == 9:
        return 1
    else:
        raise ValueError("D should end in 1, 3, 7, or 9")

def generalized_divisibility_check(N, D):
    m = find_m(D)
    
    while N >= 10:
        t = N // 10  # All digits except the last one
        q = N % 10   # Last digit
        N = m * q + t  # Transform the number
    
    return N % D == 0

def check_if_D_is_prime(D):
    if D < 2:
        return False
    if divisible_by_2(D) or divisible_by_3(D) or divisible_by_5(D):
        return False
    if divisible_by_7(D):
        return False
    for i in range(11, int(D ** 0.5) + 1, 2):
        if generalized_divisibility_check(D, i):
            return False
    return True

primes = []

for i in range(2, 1_000_000):  # Start from 2 since 1 is not prime
    if divisible_by_2(i):
        continue
    
    if divisible_by_3(i):
        continue

    if divisible_by_5(i):
        continue

    if divisible_by_7(i):
        continue

    # Generalized divisibility rule for D that is prime
    is_prime = True
    for D in range(11, int(i/2), 2):
        if check_if_D_is_prime(D):
            if generalized_divisibility_check(i, D):
                is_prime = False
                break

    if is_prime:
        primes.append(i)

# Printing the first 30 and last 30 prime numbers
print("First 30 primes:")
print(primes[:30])

print("\nLast 30 primes:")
print(primes[-30:])

print(f"\nNumber of primes found: {len(primes)}")