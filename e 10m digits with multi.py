import multiprocessing as mp
import mpmath
import time

def compute_e_segment(start, end):
    mpmath.mp.dps = end
    e_str = str(mpmath.nstr(mpmath.e, end))
    return e_str[start:end]

def combine_results(results):
    return ''.join(results)

if __name__ == '__main__':
    num_digits = 10000000  # Adjust the number of digits as required
    num_processes = min (60, mp.cpu_count())
    segment_size = num_digits // num_processes
    precision = num_digits + 10  # Extra precision to handle boundary errors

    start_time = time.time()

    with mp.Pool(processes=num_processes) as pool:
        tasks = [(i * segment_size, (i + 1) * segment_size) for i in range(num_processes)]
        results = pool.starmap(compute_e_segment, tasks)
        e_digits = combine_results(results)

    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time} seconds")
    print(e_digits[:100])  # Print the first 100 digits of e