import itertools

def simulate_game():
    # Initialize the bags with stones from 1 to 40
    bags = [i for i in range(1, 41)]
    used_pairs = set()
    moves = 0

    # Precompute all possible unique pairs (i < j)
    all_pairs = list(itertools.combinations(range(40), 2))

    # To maximize the number of moves, we'll select pairs with the smallest valid difference first
    # This approach preserves larger differences for later moves
    all_pairs_sorted = sorted(all_pairs, key=lambda pair: abs(bags[pair[0]] - bags[pair[1]]))

    # Iterate through the sorted pairs
    for pair in all_pairs_sorted:
        i, j = pair
        # Check if the pair hasn't been used and difference is at least 2
        if pair not in used_pairs:
            if abs(bags[i] - bags[j]) >= 2:
                # Determine which bag has more stones
                if bags[i] > bags[j]:
                    high, low = i, j
                else:
                    high, low = j, i

                # Perform the move: transfer 1 stone from high to low
                bags[high] -= 1
                bags[low] += 1
                moves += 1

                # Mark this pair as used
                used_pairs.add(pair)

    # After iterating through all pairs, check if additional moves are possible
    # This step ensures that any remaining valid pairs (not covered in the initial sorted list) are processed
    # Though with the sorted approach, this is typically unnecessary
    # Included here for completeness

    while True:
        # Generate all possible unique pairs that haven't been used yet
        possible_pairs = []
        for pair in all_pairs:
            if pair not in used_pairs and abs(bags[pair[0]] - bags[pair[1]]) >= 2:
                possible_pairs.append(pair)

        if not possible_pairs:
            break  # No more valid moves

        # Select the pair with the smallest difference first
        possible_pairs.sort(key=lambda pair: abs(bags[pair[0]] - bags[pair[1]]))
        selected_pair = possible_pairs[0]
        i, j = selected_pair

        # Determine which bag has more stones
        if bags[i] > bags[j]:
            high, low = i, j
        else:
            high, low = j, i

        # Perform the move
        bags[high] -= 1
        bags[low] += 1
        moves += 1

        # Mark this pair as used
        used_pairs.add(selected_pair)

    return moves, bags

def main():
    total_moves, final_bags = simulate_game()
    print(f"Total Gold Coins Collected: {total_moves}")

    # Optional: Display the final distribution of stones
    # Uncomment the following lines to see the final state of the bags
    # print("\nFinal distribution of stones in bags:")
    # for idx, stones in enumerate(final_bags, start=1):
    #     print(f"Bag {idx}: {stones} stones")

    # Verify final configuration
    count_20 = final_bags.count(20)
    count_21 = final_bags.count(21)
    print(f"\nFinal Configuration:")
    print(f"Number of bags with 20 stones: {count_20}")
    print(f"Number of bags with 21 stones: {count_21}")

if __name__ == "__main__":
    main()