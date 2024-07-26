import sympy
from mpmath import mp
import time

# e sayısının 10 milyon basamağını hesapla
mp.dps = 10000001  # Basamak sayısını belirt
e_str = str(mp.e)[2:]  # '2.' kısmını çıkart

# e sayısının ilk 10 milyon basamağını yazdır
print(f"e'nin ilk 10 milyon basamağı hesaplandı.")

# 10 basamaklı asal sayıları ve indekslerini saklamak için liste
prime_indices = []

# Verilen aralıktaki sayının asal olup olmadığını kontrol eden fonksiyon
def find_primes():
    for i in range(len(e_str) - 9):
        num = e_str[i:i + 10]
        if len(num) == 10 and num[0] != '0' and sympy.isprime(int(num)):
            prime_indices.append((i, num))

# Çalışma süresini ölçmek için başlangıç zamanı
start_time = time.time()

# Asal sayıları bul
find_primes()

# Çalışma süresini ölçmek için bitiş zamanı
end_time = time.time()

# Bulunan asal sayıları ve indeksleri yazdır
for index, prime in prime_indices:
    print(f"Index: {index}, Prime: {prime}")

# Toplam çalışma süresini yazdır
print(f"Toplam çalışma süresi: {end_time - start_time:.2f} saniye")