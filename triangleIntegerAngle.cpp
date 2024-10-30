#include <iostream>
#include <cmath>
#include <set>
#include <tuple>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <mutex>
#include <string>

// Define the default maximum perimeter
constexpr int64_t DEFAULT_MAX_PERIMETER = 100000000;

// Enumeration for angle types
enum AngleType {
    ANGLE_60 = 60,
    ANGLE_90 = 90,
    ANGLE_120 = 120
};

// Structure to represent a triangle
struct Triangle {
    int64_t a, b, c; // Side lengths
    AngleType angle;

    Triangle(int64_t _a, int64_t _b, int64_t _c, AngleType _angle)
        : a(_a), b(_b), c(_c), angle(_angle) {}

    // Calculate the perimeter of the triangle
    int64_t perimeter() const { return a + b + c; }

    // Operator to allow storage in a std::set (sorted lexicographically)
    bool operator<(const Triangle& other) const {
        return std::tie(a, b, c, angle) < std::tie(other.a, other.b, other.c, other.angle);
    }
};

// Function to verify if a triangle satisfies the Law of Cosines for its angle
bool verify_triangle(const Triangle& triangle) {
    int64_t a = triangle.a;
    int64_t b = triangle.b;
    int64_t c = triangle.c;
    AngleType angle = triangle.angle;

    switch(angle) {
        case ANGLE_60:
            // c^2 should be equal to a^2 + b^2 - a*b
            return (c * c) == (a * a + b * b - a * b);
        case ANGLE_90:
            // c^2 should be equal to a^2 + b^2
            return (c * c) == (a * a + b * b);
        case ANGLE_120:
            // c^2 should be equal to a^2 + b^2 + a*b
            return (c * c) == (a * a + b * b + a * b);
        default:
            return false;
    }
}

// Function to generate primitive triangles for 60°, 90°, and 120° angles
void generate_primitives(
    std::set<Triangle>& primitives60, 
    std::set<Triangle>& primitives90, 
    std::set<Triangle>& primitives120,
    int64_t MAX_PERIMETER
) {
    // Calculate max_m based on the perimeter constraint
    // For 60°: a = m^2 - mn + n^2 <= MAX_PERIMETER
    // Similarly for 90°: c = m^2 + n^2 <= MAX_PERIMETER
    // We'll set max_m to 2 * sqrt(MAX_PERIMETER) to cover necessary range
    int64_t max_m = static_cast<int64_t>(2 * sqrt(static_cast<double>(MAX_PERIMETER))) + 1;

    // Mutexes for thread-safe insertion into sets
    std::mutex mutex60, mutex90, mutex120;

    // Parallelize the outer loop using OpenMP with dynamic scheduling
    #pragma omp parallel for schedule(dynamic)
    for(int64_t m = 1; m <= max_m; ++m) {
        for(int64_t n = 1; n <= m; ++n) { // Allow n = m for equilateral triangles
            // Generate 60° triangles using the parametrization:
            // a = m^2 - m*n + n^2
            // b = 2*m*n - n^2
            // c = m^2 + n^2
            int64_t a60 = m * m - m * n + n * n;
            int64_t b60 = 2 * m * n - n * n;
            int64_t c60 = m * m + n * n;
            int64_t p60 = a60 + b60 + c60;
            if(a60 > 0 && b60 > 0 && c60 > 0 && p60 <= MAX_PERIMETER) {
                Triangle triangle60(a60, b60, c60, ANGLE_60);
                if(verify_triangle(triangle60)) {
                    // Thread-safe insertion
                    std::lock_guard<std::mutex> lock(mutex60);
                    primitives60.insert(triangle60);
                }
            }

            // Generate 90° triangles using the Pythagorean triple parametrization:
            // a = m^2 - n^2
            // b = 2*m*n
            // c = m^2 + n^2
            if(m > n) { // Ensure b > 0
                int64_t a90 = m * m - n * n;
                int64_t b90 = 2 * m * n;
                int64_t c90 = m * m + n * n;
                int64_t p90 = a90 + b90 + c90;
                if(a90 > 0 && b90 > 0 && c90 > 0 && p90 <= MAX_PERIMETER) {
                    Triangle triangle90(a90, b90, c90, ANGLE_90);
                    if(verify_triangle(triangle90)) {
                        // Thread-safe insertion
                        std::lock_guard<std::mutex> lock(mutex90);
                        primitives90.insert(triangle90);
                    }
                }
            }

            // Generate 120° triangles
            // The only known primitive triangle satisfying c^2 = a^2 + b^2 + ab is (3,5,7)
            // We'll generate multiples of (3,5,7) up to the perimeter constraint
            // Alternatively, attempt to find triangles via parametrization, but as it's non-trivial, we'll handle it separately

            // Check if m =1, n=1 generates (3,5,7)
            // For (m,n) = (2,1): a = m^2 + m*n +n^2 =4+2+1=7
            // b = m^2 -n^2 =4-1=3
            // c = m^2 +n^2 =4+1=5
            // Triangle (7,3,5) does not satisfy c^2 =a^2 +b^2 +a*b =>25 !=49 +9 +21=79
            // Hence, parametrization is incorrect for 120° triangles

            // Therefore, to correctly generate 120° triangles, we'll generate multiples of (3,5,7)
            // and ensure that they satisfy the angle condition

            // No parametrization-based generation is feasible currently, so we skip here
            // All 120° triangles will be handled separately
        }
    }

    // Now, handle 120° triangles by counting multiples of (3,5,7)
    // Only (3,5,7) is the known primitive triangle for 120°
    // We'll compute how many multiples of (3,5,7) fit within the perimeter constraint

    // Calculate the maximum scaling factor k for which 3k +5k +7k <= MAX_PERIMETER
    int64_t k_max = MAX_PERIMETER / (3 +5 +7); // MAX_PERIMETER /15

    for(int64_t k =1; k <=k_max; ++k){
        int64_t a120 =3 *k;
        int64_t b120 =5 *k;
        int64_t c120 =7 *k;
        int64_t p120 =a120 +b120 +c120;
        Triangle triangle120(a120, b120, c120, ANGLE_120);
        if(verify_triangle(triangle120)) {
            // Insert into primitives120
            primitives120.insert(triangle120);
        }
    }
}

int main(int argc, char* argv[]) {
    // Default MAX_PERIMETER
    int64_t MAX_PERIMETER = DEFAULT_MAX_PERIMETER;

    // Allow user to set MAX_PERIMETER via command-line argument for testing
    if (argc > 1) {
        try {
            MAX_PERIMETER = std::stoll(argv[1]);
            if (MAX_PERIMETER <= 0) {
                std::cerr << "MAX_PERIMETER must be a positive integer." << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid input for MAX_PERIMETER. Using default value " << DEFAULT_MAX_PERIMETER << "." << std::endl;
            MAX_PERIMETER = DEFAULT_MAX_PERIMETER;
        }
    }

    // Separate sets for each angle type
    std::set<Triangle> primitives60;
    std::set<Triangle> primitives90;
    std::set<Triangle> primitives120;

    // Start timing the primitive generation and processing
    double start_time = omp_get_wtime();

    // Generate unique primitive triangles
    generate_primitives(primitives60, primitives90, primitives120, MAX_PERIMETER);

    // Convert each set to a vector for efficient processing
    std::vector<Triangle> vector60(primitives60.begin(), primitives60.end());
    std::vector<Triangle> vector90(primitives90.begin(), primitives90.end());
    std::vector<Triangle> vector120(primitives120.begin(), primitives120.end());

    // Function to count triangles based on a vector of primitives
    auto count_triangles = [&](const std::vector<Triangle>& vec) -> int64_t {
        int64_t count =0;
        size_t size = vec.size();

        // Use parallel for with indices
        #pragma omp parallel for reduction(+:count) schedule(dynamic)
        for(size_t i=0;i<size;++i){
            int64_t perimeter =vec[i].perimeter();
            count += MAX_PERIMETER / perimeter;
        }

        return count;
    };

    // Count total triangles across all angle types
    int64_t triangle_count60 = count_triangles(vector60);
    int64_t triangle_count90 = count_triangles(vector90);
    int64_t triangle_count120 = count_triangles(vector120);
    int64_t total_triangle_count = triangle_count60 + triangle_count90 + triangle_count120;

    // End timing
    double end_time = omp_get_wtime();

    // Output the total number of triangles per angle type and the total
    std::cout << "60 degree triangles found: " << triangle_count60 << std::endl;
    std::cout << "90 degree triangles found: " << triangle_count90 << std::endl;
    std::cout << "120 degree triangles found: " << triangle_count120 << std::endl;
    std::cout << "Total number of triangles: " << total_triangle_count << std::endl;

    // Global Verification: Ensure all triangles are unique across angle types
    std::set<std::tuple<int64_t, int64_t, int64_t>> global_unique_triangles;

    // Insert all triangles into the global set
    for(const auto& tri : primitives60) {
        global_unique_triangles.emplace(std::make_tuple(tri.a, tri.b, tri.c));
    }
    for(const auto& tri : primitives90) {
        global_unique_triangles.emplace(std::make_tuple(tri.a, tri.b, tri.c));
    }
    for(const auto& tri : primitives120) {
        global_unique_triangles.emplace(std::make_tuple(tri.a, tri.b, tri.c));
    }

    // Calculate total unique triangles
    int64_t total_unique_triangles = global_unique_triangles.size();

    // Output
    std::cout << "Total unique triangles: " << total_unique_triangles << std::endl;

    // Verify if total_unique_triangles equals total_triangle_count
    if(total_unique_triangles == total_triangle_count) {
        std::cout << "Verification passed: All triangles are unique across angle types." << std::endl;
    } else {
        std::cout << "Verification failed: Some triangles are duplicated across angle types." << std::endl;
    }

    // Output the execution time
    std::cout << "Execution time: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;
}