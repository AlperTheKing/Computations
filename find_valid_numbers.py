from itertools import product

def is_divisible_by_21(number):
    return number % 21 == 0

def find_valid_numbers():
    digits = ['3', '7']
    valid_numbers = []
    
    # Iterate over all possible 7-digit numbers with digits 3 and 7
    for number_tuple in product(digits, repeat=7):
        number_str = ''.join(number_tuple)
        number = int(number_str)
        
        if is_divisible_by_21(number):
            valid_numbers.append(number)
    
    return valid_numbers

valid_numbers = find_valid_numbers()

print(f"Toplam {len(valid_numbers)} sayı bulunmuştur.")
print("Bu sayılar:")
for number in valid_numbers:
    print(number)
