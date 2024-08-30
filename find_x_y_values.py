import math

# Pozitif tam sayılar için belirli bir aralık seçiyoruz
max_x = 1000  # x için maksimum değer
max_y = 1000  # y için maksimum değer

# Çözüm bulma fonksiyonu
def find_solutions(max_x, max_y):
    solutions = []
    
    for x in range(1, max_x + 1):  # x 1'den max_x'e kadar artar
        for y in range(1, max_y + 1):  # y 1'den max_y'e kadar artar
            lhs = 4**x + 3**y  # Sol tarafı hesapla: 4^x + 3^y
            z = math.isqrt(lhs)  # LHS'nin karekökünü hesapla
            
            if z**2 == lhs:  # Kareköklü sayı gerçekten tam kare mi?
                solutions.append((x, y, z))  # Eğer öyleyse, çözümü kaydet
    
    return solutions

# Çözümleri bul
solutions = find_solutions(max_x, max_y)

# Çözümleri yazdır
if solutions:
    print(f"Toplam {len(solutions)} çözüm bulundu:")
    for solution in solutions:
        print(f"x = {solution[0]}, y = {solution[1]}, z = {solution[2]}")
else:
    print("Hiçbir çözüm bulunamadı.")
