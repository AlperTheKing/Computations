import sympy
from mpmath import mp
import threading
import time

# e sayısının 10 milyon basamağını hesapla
mp.dps = 10000001  # Basamak sayısını belirt
e_str = str(mp.e)[2:]  # '2.' kısmını çıkart

# e sayısının ilk 10 milyon basamağını yazdır
print(f"e'nin ilk 10 milyon basamağı hesaplandı.")

# 10 basamaklı asal sayıları ve indekslerini saklamak için liste
prime_indices = []
lock = threading.Lock()  # Listeye erişimi senkronize etmek için kilit

# Verilen aralıktaki sayının asal olup olmadığını kontrol eden fonksiyon
def find_primes(start, end):
    local_primes = []
    for i in range(start, end):
        num = e_str[i:i + 10]
        if len(num) == 10 and num[0] != '0' and sympy.isprime(int(num)):
            local_primes.append((i, num))
    with lock:
        prime_indices.extend(local_primes)

# Thread fonksiyonu
def thread_function(start, step):
    end = min(start + step, len(e_str) - 9)
    find_primes(start, end)

# Çalışma süresini ölçmek için başlangıç zamanı
start_time = time.time()

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

# Çalışma süresini ölçmek için bitiş zamanı
end_time = time.time()

# Bulunan asal sayıları ve indeksleri yazdır
for index, prime in prime_indices:
    print(f"Index: {index}, Prime: {prime}")

# Toplam çalışma süresini yazdır
print(f"Toplam çalışma süresi: {end_time - start_time:.2f} saniye")