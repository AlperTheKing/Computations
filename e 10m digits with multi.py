import multiprocessing as mp
import mpmath
import time

def compute_e_segment(start, end):
    return str(mpmath.nstr(mpmath.e, end))[:end][-1*(end-start):]

def combine_results(results):
    return ''.join(results)

if __name__ == '__main__':
    mpmath.mp.dps = 10000000  # Set decimal places for e
    num_processes = mp.cpu_count()
    segment_size = 10000000 // num_processes
    
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        tasks = [(i * segment_size, (i + 1) * segment_size) for i in range(num_processes)]
        results = pool.starmap(compute_e_segment, tasks)
        e_digits = combine_results(results)

    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time} seconds")
    print(e_digits[:100])  # Print the first 100 digits of e
