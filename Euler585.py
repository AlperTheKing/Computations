import sympy
from sympy import sqrt, simplify
from sympy.simplify.sqrtdenest import sqrtdenest
from multiprocessing import Pool, cpu_count
import time
import math

def is_perfect_square(n):
    """Check if a number is a perfect square."""
    return int(math.isqrt(n))**2 == n

def generate_non_perfect_squares(limit):
    """Generate a list of non-perfect square integers up to a limit."""
    return [i for i in range(1, limit+1) if not is_perfect_square(i)]

def process_x(args):
    """Process a single value of x to find valid denested expressions."""
    x, y_values, z_values = args
    unique_denested = set()
    for y in y_values:
        for z in z_values:
            expr = sqrt(x + sqrt(y) + sqrt(z))
            try:
                # Attempt to denest the expression
                denested_expr = sqrtdenest(expr)
                denested_expr = simplify(denested_expr)
                # Check if the denested expression is a sum/difference of square roots
                if denested_expr.is_Add or denested_expr.is_Number:
                    valid = True
                    for term in denested_expr.as_ordered_terms():
                        coeff, radical = term.as_coeff_Mul()
                        if coeff not in [1, -1]:
                            valid = False
                            break
                        if not (radical.is_Pow and radical.args[1] == sympy.S(1)/2 and radical.args[0].is_Integer):
                            valid = False
                            break
                    if valid:
                        value = denested_expr.evalf()
                        unique_denested.add(round(float(value), 10))
            except Exception:
                continue
    return unique_denested

def compute_F(n, L):
    """Compute F(n) using multiprocessing."""
    start_time = time.time()
    num_processes = cpu_count()
    print(f"\nComputing F({n}) using {num_processes} processes with L = {L}.")
    # Generate y and z values
    y_values = generate_non_perfect_squares(L)
    z_values = y_values  # Since y and z have the same conditions
    x_values = list(range(1, n+1))
    # Prepare arguments for each process
    args = [(x, y_values, z_values) for x in x_values]
    unique_denested = set()
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for dynamic load balancing
        results = pool.imap_unordered(process_x, args)
        for res in results:
            unique_denested.update(res)
    F_n = len(unique_denested)
    end_time = time.time()
    print(f"F({n}) = {F_n}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return F_n

if __name__ == '__main__':
    n_values = [10, 15, 20, 30, 100]
    L_values = [100, 150, 200, 300, 1000]  # Set L proportional to n
    for n, L in zip(n_values, L_values):
        compute_F(n, L)