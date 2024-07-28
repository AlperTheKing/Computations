from sympy import bernoulli, binomial
import multiprocessing
import os

def calculate_term(args):
    k, n, p = args
    binomial_coeff = binomial(p + 1, k)
    bernoulli_num = bernoulli(k)
    return binomial_coeff * bernoulli_num * (n ** (p + 1 - k))

def faulhabers_formula(n, p):
    # Prepare arguments for multiprocessing
    args = [(k, n, p) for k in range(p + 1)]
    
    num_cores = os.cpu_count()  # Get the number of available CPU cores
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(calculate_term, args)
        
    sum_p = sum(results)
    return sum_p / (p + 1)

if __name__ == "__main__":
    n = int(input("Enter the last integer: "))  # upper limit
    p = int(input("Enter the power: "))   # power
    result = faulhabers_formula(n, p)
    print(f"The sum of the first {n} integers raised to the power {p} is: {result}")
