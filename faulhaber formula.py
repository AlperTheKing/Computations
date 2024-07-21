from sympy import bernoulli

def faulhabers_formula(n, p):
    # Calculate the sum of the first n integers raised to the power p
    sum_p = 0
    for k in range(p + 1):
        binomial_coeff = binomial(p + 1, k)
        bernoulli_num = bernoulli(k)
        sum_p += binomial_coeff * bernoulli_num * (n ** (p + 1 - k))

    return sum_p / (p + 1)

def binomial(n, k):
    if k == 0 or k == n:
        return 1
    if k > n:
        return 0
    if k > n - k:
        k = n - k
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

# Example usage
n = input("Enter the last integer: ")  # upper limit
p = input("Enter the power: ")   # power
result = faulhabers_formula(int(n), int(p))
print(f"The sum of the first {n} integers raised to the power {p} is: {result}")
