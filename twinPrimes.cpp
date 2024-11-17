#include <gmp.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdlib> 
#include <sstream>
#include <fstream>
#include <mutex>

// Mutex for synchronized console output
std::mutex cout_mutex;

// Function to execute ECPP tool and check primality
bool isPrimeECPP(const std::string& number_str) {
    // Path to the ECPP tool (e.g., Primo)
    // Replace "primo" with the actual command or path to your ECPP tool
    std::string command = "/home/alper/Downloads/primo-433-lx64/primo " + number_str + " > ecpp_result.txt 2>&1";

    // Execute the command
    int ret = system(command.c_str());

    // Open the result file and parse the output
    std::ifstream result_file("ecpp_result.txt");
    if (!result_file.is_open()) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Failed to open ECPP result file.\n";
        return false;
    }

    std::string line;
    bool is_prime = false;
    while (std::getline(result_file, line)) {
        // Parse the output based on the ECPP tool's format
        // Adjust the parsing logic according to your tool's output
        if (line.find("is prime") != std::string::npos) {
            is_prime = true;
            break;
        } else if (line.find("is composite") != std::string::npos) {
            is_prime = false;
            break;
        }
    }

    result_file.close();

    // Clean up the result file
    std::remove("ecpp_result.txt");

    return is_prime;
}

// Function to search for twin primes in a given range
void findTwinPrimes(mpz_t start, mpz_t end, std::atomic<long>& count) {
    mpz_t p, p_plus_2;
    mpz_init(p);
    mpz_init(p_plus_2);

    for (mpz_set(p, start); mpz_cmp(p, end) <= 0; mpz_add_ui(p, p, 1)) {
        // Skip even numbers greater than 2
        if (mpz_even_p(p) && mpz_cmp_ui(p, 2) != 0) {
            continue;
        }

        // Convert p to string for ECPP tool
        char* p_str_c = mpz_get_str(NULL, 10, p);
        std::string p_str(p_str_c);
        free(p_str_c);

        if (isPrimeECPP(p_str)) {
            // p + 2
            mpz_add_ui(p_plus_2, p, 2);
            if (mpz_cmp(p_plus_2, end) > 0) {
                break;
            }

            // Convert p + 2 to string
            char* p_plus_2_str_c = mpz_get_str(NULL, 10, p_plus_2);
            std::string p_plus_2_str(p_plus_2_str_c);
            free(p_plus_2_str_c);

            if (isPrimeECPP(p_plus_2_str)) {
                // Found a twin prime pair (p, p + 2)
                count++;

                // Optional: Print the twin prime pair
                /*
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "(" << p_str << ", " << p_plus_2_str << ")\n";
                */
            }
        }
    }

    mpz_clear(p);
    mpz_clear(p_plus_2);
}

int main(int argc, char* argv[]) {
    // Check for correct usage
    if (argc != 3) {
        std::cerr << "Usage: ./twin_primes_ecpp <start> <end>\n";
        std::cerr << "Example: ./twin_primes_ecpp 99999999999999999999999999999999999999999999999999 100000000000000000000000000000000000000000000000000\n";
        return 1;
    }

    // Initialize GMP variables for start and end
    mpz_t start, end;
    mpz_init_set_str(start, argv[1], 10);
    mpz_init_set_str(end, argv[2], 10);

    // Determine the number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2; // Fallback

    std::cout << "Using " << num_threads << " threads.\n";

    // Calculate the range size per thread
    mpz_t range_size, temp;
    mpz_init(range_size);
    mpz_init(temp);
    mpz_sub(temp, end, start);
    mpz_divexact_ui(range_size, temp, num_threads);
    mpz_clear(temp);

    // Atomic counter for twin primes
    std::atomic<long> twin_prime_count(0);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        mpz_t thread_start, thread_end;
        mpz_init_set(thread_start, start);
        mpz_addmul_ui(thread_start, range_size, i);

        if (i == num_threads - 1) {
            mpz_init_set(thread_end, end);
        } else {
            mpz_init_set(thread_end, thread_start);
            mpz_add(thread_end, thread_end, range_size);
            mpz_sub_ui(thread_end, thread_end, 1); // Exclusive end
        }

        threads.emplace_back(findTwinPrimes, thread_start, thread_end, std::ref(twin_prime_count));

        mpz_clear(thread_start);
        if (i != num_threads - 1) {
            mpz_clear(thread_end);
        }
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        th.join();
    }

    // End timing
    auto end_time_point = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time_point - start_time;

    // Output the results
    std::cout << "Total twin primes found: " << twin_prime_count.load() << "\n";
    std::cout << "Time taken: " << elapsed.count() << " seconds.\n";

    // Clear GMP variables
    mpz_clear(start);
    mpz_clear(end);
    mpz_clear(range_size);
    mpz_clear(temp);

    return 0;
}