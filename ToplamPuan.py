# jury1 = []
# jury2 = []
# jury3 = []

# # Loop over players
# for player in range(1, 7):
#     # Assign scores from the range 1 to 6
#     for i in range(1, 7):
#         jury1.append(i)
#         for j in range(1, 7):
#             if j != i:
#                 jury2.append(j)  # Ensure that jury2 picks a score different from jury1
#                 for k in range(1, 7):
#                     if k != i and k != j:
#                         jury3.append(k)  # Ensure that jury3 picks a score different from jury1 and jury2
#                         if ((len(jury1)) == 6): print(f"jury1 = {jury1}")
#                         if ((len(jury2)) == 6): print(f"jury2 = {jury2}")
#                         if ((len(jury3)) == 6): print(f"jury3 = {jury3}")


from itertools import permutations

scores = [1, 2, 3, 4, 5, 6]

# Generate all permutations and convert the iterator to a list
all_permutations = list(permutations(scores))

# Print the permutations
for perm in all_permutations:
    print(perm)