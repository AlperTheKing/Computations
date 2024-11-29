from math import isqrt
from collections import defaultdict

def count_unit_fraction_sums(n):
    """
    Returns the number of ways to write 1/n as a sum of two unit fractions,
    along with the specific pairs (a, b).
    """
    n_squared = n * n
    divisors = set()
    
    # Find all divisors of n^2
    for i in range(1, isqrt(n_squared) + 1):
        if n_squared % i == 0:
            divisors.add(i)
            divisors.add(n_squared // i)
    
    # Sort the divisors
    sorted_divisors = sorted(divisors)
    
    # Find valid (a, b) pairs
    pairs = []
    for x in sorted_divisors:
        y = n_squared // x
        if x > y:
            continue  # Ensure a <= b
        a = x + n
        b = y + n
        pairs.append((a, b))
    
    return len(pairs), pairs

# Part (a)
print("Part (a):")
for n in [5, 7]:
    count, pairs = count_unit_fraction_sums(n)
    print(f"1/{n} can be expressed in {count} way(s):")
    for a, b in pairs:
        print(f"  1/{n} = 1/{a} + 1/{b}")
    print()

# Part (b)
print("Part (b):")
for n in [4, 6]:
    count, pairs = count_unit_fraction_sums(n)
    print(f"1/{n} can be expressed in {count} way(s):")
    for a, b in pairs:
        print(f"  1/{n} = 1/{a} + 1/{b}")
    print()