import multiprocessing as mp
from sympy import factorial, isprime

def is_factorial_plus_one_prime(n):
    """n! + 1 sayısının asal olup olmadığını kontrol eder."""
    number = factorial(n) + 1
    if isprime(number):
        return (n, number)
    return None

def parallel_search(limit, num_processes=mp.cpu_count()):
    """Çoklu çekirdek kullanarak n! + 1 şeklinde ifade edilen asal sayıları arar."""
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(is_factorial_plus_one_prime, range(1, limit + 1))
    
    # Asal olanları filtreleyin
    primes = [result for result in results if result is not None]
    return primes

if __name__ == "__main__":
    limit = 1000  # Aranacak n değerinin üst sınırı
    primes = parallel_search(limit)
    
    if primes:
        for n, prime in primes:
            print(f"{n}! + 1 = {prime} (asal)")
    else:
        print("Belirtilen aralıkta n! + 1 şeklinde ifade edilebilen asal sayı bulunamadı.")