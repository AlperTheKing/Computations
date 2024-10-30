#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <numeric>
#include <iomanip> // For std::setw

// Constants
constexpr int MAX_PERIMETER = 10'000'000;
std::atomic<int> primitive_triangle_count{0};
std::atomic<int> current_a{1}; // Shared counter for the `a` side
std::atomic<int> progress{0};  // Tracks progress for perimeter updates

// Helper function to compute the GCD of three integers
int gcd(int a, int b, int c) {
    return std::gcd(std::gcd(a, b), c);
}

// Check if (a, b, c) forms a valid primitive triangle
bool is_primitive_triangle(int a, int b, int c) {
    return gcd(a, b, c) == 1;
}

// Worker function for each thread
void find_primitive_triangles() {
    int a;
    while ((a = current_a.fetch_add(1)) <= MAX_PERIMETER / 3) {
        for (int b = a; a + b < MAX_PERIMETER; ++b) {
            int c = a + b - 1; // Set c to its minimum feasible value
            while (a + b + c <= MAX_PERIMETER) {
                int perimeter = a + b + c;
                
                // Only check the necessary triangle inequality a + b > c
                if (a + b > c && is_primitive_triangle(a, b, c)) {
                    primitive_triangle_count++;
                    
                    // Debugging: Print details of the triangle found
                    std::cout << "Triangle found: (" << a << ", " << b << ", " << c << "), Perimeter: " << perimeter << "\n";
                }
                
                // Update progress based on perimeter
                int percent_complete = (perimeter * 100) / MAX_PERIMETER;
                int expected_progress = percent_complete - 1;
                if (progress.compare_exchange_strong(expected_progress, percent_complete)) {
                    // Print the progress bar
                    std::cout << "\rProgress: [";
                    int bar_width = 50; // Width of the progress bar
                    int pos = bar_width * percent_complete / 100;
                    for (int i = 0; i < bar_width; ++i) {
                        if (i < pos) std::cout << "=";
                        else if (i == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::setw(3) << percent_complete << "%";
                    std::cout.flush();
                }

                c++;
            }
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(find_primitive_triangles);
    }

    // Join threads
    for (auto &t : threads) {
        t.join();
    }

    // Print a newline after the progress bar completes
    std::cout << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Print the result after all threads complete
    std::cout << "Number of primitive triangles: " << primitive_triangle_count.load() << "\n";
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    return 0;
}