with open("primes1T.txt", "rb") as f:
    f.seek(-130,2)
    last_part = f.read(130)
    print(int(last_part.decode()))