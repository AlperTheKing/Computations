def find_max_subset():
    numbers = list(range(2, 22))  # Numbers from 2 to 21
    subset = []

    # Loop through numbers and build the subset
    for num in numbers:
        is_divisible = False
        for selected in subset:
            if num % selected == 0 or selected % num == 0:
                is_divisible = True
                break
        if not is_divisible:
            subset.append(num)

    return subset


# Find the maximum subset
max_subset = find_max_subset()

# Output the result
print(f"Maximum subset: {max_subset}")
print(f"Size of the subset: {len(max_subset)}")