import multiprocessing

# Function to check the triplet condition
def check_triplet(a, b, c):
    term = (8 * (a ** 3)) + (15 * (a ** a)) + (6 * a) - (27 * (b ** 2) * c)
    if term == 1:
        return (a, b, c)
    return None

# Function to process a range of values for a
def process_range(a, r):
    results = []
    for b in range(r):
        for c in range(r):
            result = check_triplet(a, b, c)
            if result:
                results.append(result)
    return results

if __name__ == '__main__':
    r = 1000
    count = 0
    found_triplets = []

    # Define the pool of workers (using all available CPUs)
    with multiprocessing.Pool() as pool:
        # Map the function `process_range` over the range of a values
        results = pool.starmap(process_range, [(a, r) for a in range(r)])
    
    # Flatten the list of results and count the triplets
    for sublist in results:
        found_triplets.extend(sublist)
        count += len(sublist)
    
    # Print all found triplets
    for triplet in found_triplets:
        print(triplet)

    print(f"{count} triplet(s) found")