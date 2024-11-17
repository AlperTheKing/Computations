#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional> // For std::reference_wrapper
#include <numeric>    // For std::gcd
#include <string>

// Type definition for 128-bit integers
typedef __int128 int128;

// Mutex for thread-safe operations
std::mutex cout_mutex;

// Helper function to convert __int128 to string for output
std::string int128_to_string(int128 n) {
    bool negative = false;
    if (n < 0) {
        negative = true;
        n = -n;
    }
    std::string s = "";
    if (n == 0) return "0";
    while (n > 0) {
        char digit = '0' + (n % 10);
        s = digit + s;
        n /= 10;
    }
    if (negative) {
        s = "-" + s;
    }
    return s;
}

// Simple prime checking function
bool is_prime(int n) {
    if (n < 2)
        return false;
    if (n == 2 || n == 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    int sqrt_n = static_cast<int>(sqrt(n));
    for(int i = 5; i <= sqrt_n; i += 6){
        if(n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

// Function to factorize n and return its prime factors
std::vector<int> prime_factors(int n){
    std::vector<int> factors;
    if(n < 1){
        return factors;
    }
    while(n % 2 == 0){
        factors.push_back(2);
        n /= 2;
    }
    while(n % 3 == 0){
        factors.push_back(3);
        n /= 3;
    }
    int i = 5;
    int w = 2;
    while(i * i <= n){
        while(n % i == 0){
            factors.push_back(i);
            n /= i;
        }
        i += w;
        w = 6 - w;
    }
    if(n > 1){
        factors.push_back(n);
    }
    return factors;
}

// Function to check if all prime factors are congruent to 1 mod 8
bool all_factors_congruent_1_mod8(int n){
    if(n == 0){
        return false;
    }
    std::vector<int> factors = prime_factors(n);
    for(auto &p : factors){
        if(p % 8 != 1){
            return false;
        }
    }
    return true;
}

// Function to solve the conic equation (3b) for x and y
// (2m^2 + n^2) y^2 = -(6m^2 -8mn +3n^2) x^2 -2(2m^2 -n^2)x -2mn
// This function searches for integer solutions within a specified range
bool solve_conic(int m, int n, int &x_sol, int &y_sol){
    // Coefficients
    int128 A = -(6LL * m * m - 8LL * m * n + 3LL * n * n);
    int128 B = -2LL * (2LL * m * m - n * n);
    int128 C = -2LL * m * n;
    int128 D = 2LL * m * m + n * n;

    // Define the search range for x
    // Adjust the range based on expected solution sizes
    for(int x = -100; x <= 100; ++x){ // Reduced range for quicker testing
        // Calculate the right-hand side (RHS) of the equation
        int128 rhs = A * x * x + B * x + C;
        if(rhs <= 0){
            continue; // y^2 must be positive
        }
        // Check if RHS is divisible by D
        if(rhs % D != 0){
            continue;
        }
        int128 y_sq = rhs / D;
        // Check if y_sq is a perfect square
        double y_d = sqrt(static_cast<double>(y_sq));
        int128 y = static_cast<int128>(round(y_d));
        if(y * y == y_sq){
            x_sol = x;
            y_sol = static_cast<int>(y);
            return true;
        }
    }
    return false;
}

// Function to process a range of m values
void process_range(int m_start, int m_end, int n_max, std::vector<std::vector<long long>> &solutions){
    for(int m = m_start; m <= m_end; ++m){
        for(int n = -n_max; n <= n_max; n += 2){ // n is odd, from -20 to +20
            if(n == 0){
                continue; // Avoid division by zero
            }
            if(std::gcd(m, n) !=1){
                continue;
            }
            int128 val1 = 2LL * m * m + n * n;
            int128 val2 = 2LL * m * m -4LL * m * n + n * n;
            if(val1 <=0 || val2 <=0){
                continue;
            }
            if(!all_factors_congruent_1_mod8(static_cast<int>(val1))){
                continue;
            }
            if(!all_factors_congruent_1_mod8(static_cast<int>(val2))){
                continue;
            }
            // Solve the conic equation for x and y
            int x, y;
            bool has_solution = solve_conic(m, n, x, y);
            if(has_solution){
                // From equation (3c): (2m^2 +n^2) t =4(2m^2 -n^2)x^2 +8mnx + (n^2 -2m^2)
                int128 numerator = 4LL * (2LL * m * m - n * n) * x * x + 8LL * m * n * x + (n * n - 2LL * m * m);
                int128 denominator = 2LL * m * m + n * n;
                if(denominator ==0){
                    continue;
                }
                if(numerator % denominator !=0){
                    continue;
                }
                int128 t = numerator / denominator;
                // From equation (2a): r = x + y, s = x - y
                int128 r = static_cast<int128>(x) + static_cast<int128>(y);
                int128 s = static_cast<int128>(x) - static_cast<int128>(y);
                // Compute the greatest common divisor of r, s, t
                int128 g = std::gcd(std::gcd(static_cast<long long>(r), static_cast<long long>(s)), static_cast<long long>(t));
                if(g ==0){
                    continue;
                }
                // Scale down r, s, t by gcd to ensure they're coprime
                r /= g;
                s /= g;
                t /= g;
                // Now, compute A^4 + B^4 + C^4
                int128 lhs = r * r * r * r + s * s * s * s + t * t * t * t;
                if(lhs <=0){
                    continue;
                }
                // Compute the fourth root of lhs to find D
                double D_double = pow(static_cast<double>(lhs), 0.25);
                int128 D = static_cast<int128>(round(D_double));
                if(D <=0 || D > 1000000){
                    continue;
                }
                if(D * D * D * D != lhs){
                    continue;
                }
                // Ensure A <= B <= C
                if(!(r <= s && s <= t)){
                    continue;
                }
                // Lock the mutex before writing to solutions
                {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    solutions.emplace_back(std::vector<long long>{
                        static_cast<long long>(r),
                        static_cast<long long>(s),
                        static_cast<long long>(t),
                        static_cast<long long>(D)
                    });
                }
            }
        }
    }
}

int main(){
    // Parameters
    int m_min = -20;
    int m_max = 20;
    int n_min = -20;
    int n_max = 20;
    int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0){
        num_threads =4; // Default to 4 if unable to detect
    }
    std::cout << "Using " << num_threads << " threads.\n";

    // Correctly define all_solutions as a vector of vector of vectors
    std::vector<std::vector<std::vector<long long>>> all_solutions(num_threads);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create and launch threads
    std::vector<std::thread> threads;
    for(int i =0;i<num_threads;i++){
        int m_start = m_min + i * ((m_max - m_min +1) / num_threads);
        int m_end = (i == num_threads -1) ? m_max : (m_min + (i +1) * ((m_max - m_min +1) / num_threads) -1);
        threads.emplace_back([m_start, m_end, n_max, &all_solutions, i]() {
            process_range(m_start, m_end, n_max, all_solutions[i]);
        });
    }

    // Join threads
    for(auto &th : threads){
        th.join();
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // Collect all solutions
    std::vector<std::vector<long long>> solutions;
    for(auto &vec : all_solutions){
        solutions.insert(solutions.end(), vec.begin(), vec.end());
    }

    // Remove duplicate solutions
    std::sort(solutions.begin(), solutions.end(), [](const std::vector<long long> &a, const std::vector<long long> &b) -> bool {
        return a < b;
    });
    solutions.erase(std::unique(solutions.begin(), solutions.end()), solutions.end());

    // Display solutions
    if(!solutions.empty()){
        std::cout << "Solutions found:\n";
        for(auto &sol : solutions){
            std::cout << int128_to_string(sol[0]) << "^4 + "
                      << int128_to_string(sol[1]) << "^4 + "
                      << int128_to_string(sol[2]) << "^4 = "
                      << int128_to_string(sol[3]) << "^4\n";
        }
    }
    else{
        std::cout << "No solutions found within the specified range.\n";
    }

    std::cout << "Total solutions found: " << solutions.size() << "\n";
    std::cout << "Time taken: " << diff.count() << " seconds\n";

    return 0;
}
