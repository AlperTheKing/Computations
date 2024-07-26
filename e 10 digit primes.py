import sympy
from mpmath import mp
import threading

# e sayısının 1000 basamağını hesapla
mp.dps = 1001  # Basamak sayısını belirt
e_str = str(mp.e)[2:]  # '2.' kısmını çıkart

# e sayısının ilk 1000 basamağını yazdır
print(f"e'nin ilk 1000 basamağı: {e_str}")

# 10 basamaklı asal sayıları ve indekslerini saklamak için liste
prime_indices = []

# Verilen aralıktaki sayının asal olup olmadığını kontrol eden fonksiyon
def find_primes(start, end):
    global prime_indices
    for i in range(start, end):
        num = e_str[i:i + 10]
        if len(num) == 10 and sympy.isprime(int(num)):
            prime_indices.append((i, num))

# Thread fonksiyonu
def thread_function(start, step):
    end = min(start + step, len(e_str) - 9)
    find_primes(start, end)

# Multi-threading
num_threads = 16
step = len(e_str) // num_threads
threads = []

# Threadleri oluştur ve başlat
for i in range(num_threads):
    start = i * step
    thread = threading.Thread(target=thread_function, args=(start, step))
    threads.append(thread)
    thread.start()

# Threadlerin bitmesini bekle
for thread in threads:
    thread.join()

# Bulunan asal sayıları ve indeksleri yazdır
for index, prime in prime_indices:
    print(f"Index: {index}, Prime: {prime}")