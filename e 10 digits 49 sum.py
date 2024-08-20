import mpmath
import multiprocessing as mp

def compute_e_to_n_digits(n):
    mpmath.mp.dps = n
    return str(mpmath.e)

def find_special_numbers(segment):
    special_numbers = []
    for i in range(len(segment) - 9):
        substring = segment[i:i + 10]
        if substring.isdigit() and sum(int(digit) for digit in substring) == 49:
            special_numbers.append(substring)
    return special_numbers

if __name__ == '__main__':
    # Calculate the first 1000 digits of e
    num_digits = 1000
    e_str = compute_e_to_n_digits(num_digits)

    # Determine segment size (e.g., 100)
    segment_size = 100
    segments = [e_str[i:i + segment_size + 9] for i in range(0, len(e_str) - 9, segment_size)]

    # Process segments using multiprocessing
    with mp.Pool(min(60, mp.cpu_count())) as pool:
        results = pool.map(find_special_numbers, segments)

    # Combine all results
    special_numbers = [number for sublist in results for number in sublist]

    # Print the results
    print(f"Found {len(special_numbers)} special numbers.")
    print(special_numbers[:10])  # Print the first 10 results as a sample