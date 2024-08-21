import time
from multiprocessing import Pool, cpu_count

def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

def calculate_collatz_range(start, end):
    local_max_steps = 0
    local_number_with_max_steps = 0
    
    for i in range(start, end):
        steps = collatz_steps(i)
        if steps > local_max_steps:
            local_max_steps = steps
            local_number_with_max_steps = i
            
    return local_number_with_max_steps, local_max_steps

def split_range(start, end, num_splits):
    step = (end - start) // num_splits
    ranges = [(start + i * step, start + (i + 1) * step) for i in range(num_splits)]
    ranges[-1] = (ranges[-1][0], end)  # Ensure the last range goes up to the actual end
    return ranges

def main():
    groups = [
        (1, 1000),
        (1000, 10_000),
        (10_000, 100_000),
        (100_000, 1_000_000),
        (1_000_000, 10_000_000),
        (10_000_000, 100_000_000),
        (100_000_000, 1_000_000_000),
        (1_000_000_000, 10_000_000_000),  
        (10_000_000_000, 100_000_000_000)
    ]

    num_processes = min(60, cpu_count())
    
    for start, end in groups:
        start_time = time.time()

        ranges = split_range(start, end, num_processes)

        with Pool(processes=num_processes) as pool:
            results = pool.starmap(calculate_collatz_range, ranges)
        
        number_with_max_steps, max_steps = max(results, key=lambda x: x[1])
        print(f"Range {start} to {end}:")
        print(f"  Number with max steps: {number_with_max_steps} ({max_steps} steps)")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Hesaplama süresi: {elapsed_time:.2f} saniye\n")

if __name__ == "__main__":
    main()