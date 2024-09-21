#include <iostream>
#include <set>
#include <tuple>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>

typedef long long ll;
std::mutex triplet_mutex;  // Mutex to protect shared triplet data

// Function to generate square divisors of a number
std::vector<ll> get_square_divisors(ll n) {
    std::vector<ll> divisors;
    for (ll i = 1; i * i <= n; ++i) {
        if (n % (i * i) == 0) {
            divisors.push_back(i);
        }
    }
    return divisors;
}

// Container to store unique triplets (using a set to eliminate duplicates)
std::set<std::tuple<ll, ll, ll>> triplets;

// Function to be executed by each thread
void find_cardano_triplets(ll start_a, ll limit, std::atomic<bool>& done) {
    ll a = start_a;
    while (!done.load()) {
        // Calculate N = 8a^3 + 15a^2 + 6a - 1
        ll a_squared = a * a;
        ll a_cubed = a_squared * a;
        ll N = 8 * a_cubed + 15 * a_squared + 6 * a - 1;

        // Check if N is divisible by 27
        if (N % 27 == 0) {
            // Calculate N / 27
            ll N27 = N / 27;

            // Get all square divisors of N / 27
            std::vector<ll> divisors = get_square_divisors(N27);

            // For each divisor b, calculate the corresponding c value
            for (ll b : divisors) {
                ll b_squared = b * b;
                ll c = N27 / b_squared;
                ll sum = a + b + c;
                if (sum <= limit) {
                    // Lock mutex to safely add the unique triplet to the shared container
                    std::lock_guard<std::mutex> guard(triplet_mutex);
                    triplets.insert(std::make_tuple(a, b, c));  // Add the triplet to the set
                } else if (a > limit) {
                    done.store(true);  // Signal other threads to stop
                    break;
                }
            }
        }
        a += 3 * std::thread::hardware_concurrency();  // Increase a by the total step size
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    ll limit = 110000000;

    // Use atomic boolean to signal threads to stop
    std::atomic<bool> done(false);

    // Get the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Default to 4 if the system cannot determine it

    std::vector<std::thread> threads;

    // Launch multiple threads with different starting points for `a`
    for (unsigned int i = 0; i < num_threads; ++i) {
        ll thread_start_a = 2 + 3 * i;  // Each thread starts with a different value of `a`
        threads.push_back(std::thread(find_cardano_triplets, thread_start_a, limit, std::ref(done)));
    }

    // Join all threads to ensure they complete execution
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Output the total number of unique triplets found
    std::cout << "Number of unique Cardano Triplets: " << triplets.size() << std::endl;

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Execution time: " << diff.count() << " seconds" << std::endl;

    return 0;
}