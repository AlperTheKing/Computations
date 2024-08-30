count = 0

for x in range(100):  # x değerlerini deniyoruz
    for y in range(100):  # y değerlerini deniyoruz
        if (x * y - 7) ** 2 == x ** 2 + y ** 2:
            print(f"x = {x}, y = {y}")
            count += 1

print(f"Toplam {count} tane çözüm var.")
