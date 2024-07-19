number = int(input("Enter a number: "))
steps = 0

for i in range(1000):
    if number == 1:
        break
    elif number % 2 == 1:
        number = 3 * number + 1
        steps += 1
    else:
        number = number / 2
        steps += 1

if number == 1:
    print("It took", steps, "steps")
else:
    print("The number didn't reach 1 yet")