import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return n, steps

def calculate_collatz_range(start, end):
    local_max_steps = 0
    local_number_with_max_steps = 0
    local_collatz_data = {}
    
    for i in range(start, end):
        number, steps = collatz_steps(i)
        local_collatz_data[i] = steps
        if steps > local_max_steps:
            local_max_steps = steps
            local_number_with_max_steps = i
            
    return local_collatz_data, local_number_with_max_steps, local_max_steps

def main():
    max_number = 10_000_000
    num_threads = 16
    chunk_size = max_number // num_threads
    futures = []
    
    max_steps = 0
    number_with_max_steps = 0
    collatz_data = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size + 1
            end = (i + 1) * chunk_size + 1
            futures.append(executor.submit(calculate_collatz_range, start, end))
        
        for future in as_completed(futures):
            local_collatz_data, local_number_with_max_steps, local_max_steps = future.result()
            collatz_data.update(local_collatz_data)
            if local_max_steps > max_steps:
                max_steps = local_max_steps
                number_with_max_steps = local_number_with_max_steps
    
    # Plotting the results
    plt.figure(figsize=(14, 7))
    keys = list(collatz_data.keys())
    values = list(collatz_data.values())
    plt.plot(keys, values, marker='o', linestyle='-', color='b', markersize=0.5)
    plt.xlabel('Number')
    plt.ylabel('Steps to reach 1')
    plt.title('Collatz Conjecture Steps from 1 to 10,000,000')
    plt.grid(True)

    # Highlight the number with the maximum steps
    plt.plot(number_with_max_steps, max_steps, marker='o', markersize=8, color='r')
    plt.text(number_with_max_steps, max_steps, f'Max steps: {max_steps}\n(Number: {number_with_max_steps})', 
             horizontalalignment='left', verticalalignment='bottom', fontsize=12, color='red')

    plt.show()

    print(f"The number with the maximum steps is {number_with_max_steps} with {max_steps} steps.")

if __name__ == "__main__":
    main()