#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Rastgele bir sayı üretecek fonksiyon (1-33 arası rastgele sayı)
int random_value(int max_value) {
    static std::mt19937 rng(std::random_device{}());  // Random number generator
    std::uniform_int_distribution<int> dist(1, max_value);  // 1 ile max_value arasında sayı
    return dist(rng);
}

// Ali Baba'nın taşları yerleştirdiği simülasyon
int place_stones(int num_chests) {
    int operations = 0;  // Ali Baba'nın toplam yaptığı işlem sayısı
    std::vector<std::pair<int, int>> used_pairs;  // Kullanılan (m, n) çiftleri
    std::vector<int> stone_requests;  // Her işlemde Ali Baba'nın istediği taş sayıları

    // Boş aralıklar oluşturuluyor
    std::vector<std::pair<int, int>> intervals = {{1, num_chests}};

    while (!intervals.empty()) {
        operations++;  // Yeni bir taş isteme işlemi
        std::vector<std::pair<int, int>> new_intervals;

        // Ali Baba'nın hırsızlardan istediği taş sayısı (1 ile 33 arasında)
        int stone_amount = random_value(33);
        stone_requests.push_back(stone_amount);  // İstenen taş miktarı kaydediliyor
        std::cout << "İşlem " << operations << ": Ali Baba " << stone_amount << " taş istedi.\n";
        std::cout << "Bu taş türü şu aralıklara yerleştirildi: ";

        // Her aralık için
        for (auto &interval : intervals) {
            int m = interval.first;
            int n = interval.second;

            // Orta noktayı bul ve bu aralığı ikiye böl
            int mid = (m + n) / 2;
            used_pairs.push_back({m, n});

            // Aralıklar yazdırılıyor
            std::cout << "(" << m << ", " << n << ") ";

            if (m < mid) {
                new_intervals.push_back({m, mid});  // Sol aralık
            }
            if (mid + 1 < n) {
                new_intervals.push_back({mid + 1, n});  // Sağ aralık
            }
        }

        std::cout << std::endl;
        intervals = new_intervals;  // Kalan aralıklar
    }

    return operations;
}

int main() {
    int num_chests = 33;

    // Zaman ölçümünü başlat
    auto start_time = std::chrono::high_resolution_clock::now();

    // Simülasyonu çalıştır
    int min_operations = place_stones(num_chests);

    // Zaman ölçümünü bitir
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Sonuçları yazdır
    std::cout << "Minimum taş türü kullanma sayısı: " << min_operations << "\n";
    std::cout << "Geçen süre: " << elapsed.count() << " saniye\n";

    return 0;
}