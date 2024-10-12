# Josephus problem simulation with adjustable starting point
def josephus_simulation(n, k, start):
    people = list(range(1, n + 1))
    people = people[start-1:] + people[:start-1]  # Rotate the list to start at a specific number
    elimination_order = []
    index = 0
    while len(people) > 2:  # Continue until only 2 people are left
        index = (index + k - 1) % len(people)  # Calculate the elimination index
        elimination_order.append(people.pop(index))  # Eliminate the person at the index
    return elimination_order, people  # Return the elimination order and remaining two people

# Define the total number of people (66) and the step count (3rd person elimination)
n = 66
k = 3

# Loop over all possible starting points from 1 to 66 and find when remaining people are 44 and 66
correct_start = None
for start in range(1, n + 1):
    elimination_order, remaining_people = josephus_simulation(n, k, start)
    if set(remaining_people) == {44, 66}:  # Check if the remaining people are 44 and 66
        correct_start = start
        print(f"Correct starting point: {correct_start}, Remaining people: {remaining_people}")
        break  # Stop once the correct starting point is found

# Now show the elimination order for the correct starting point
if correct_start:
    elimination_order, remaining_people = josephus_simulation(n, k, correct_start)

    # Print the elimination order
    print(f"\nElimination order starting from {correct_start}:")
    for i, num in enumerate(elimination_order, 1):
        print(f"{i}. eliminated: {num}")

    # Print the remaining two people
    print(f"\nRemaining two people: {remaining_people[0]} and {remaining_people[1]}")
else:
    print("No correct starting point found.")