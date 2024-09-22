#include <iostream>
#include <unordered_set>
#include <tuple>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>

typedef long long ll;
std::mutex triplet_mutex;  // Mutex to protect shared triplet data

// Custom hash function for tuple of (a, b, c)
struct hash_triplet {
    std::size_t operator()(const std::tuple<ll, ll, ll>& triplet) const {
        auto h1 = std::hash<ll>()(std::get<0>(triplet));
        auto h2 = std::hash<ll>()(std::get<1>(triplet));
        auto h3 = std::hash<ll>()(std::get<2>(triplet));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Function to find valid Cardano Triplets (a, b, c) for a range of `a` values
void find_cardano_triplets(ll start_a, ll limit, std::atomic<bool>& done, std::unordered_set<std::tuple<ll, ll, ll>, hash_triplet>& local_triplets, ll step) {
    ll a = start_a;
    while (!done.load()) {
        ll u = (2 * a - 1) / 3;
        if (u % 2 == 0) {
            a += step;
            continue;  // u must be odd
        }

        ll lhs = u * u * u + a * a;  // u^3 + a^2
        bool found_triplet_for_a = false;  // Track whether at least one triplet was found for this value of a

        for (ll b = 1; b * b <= lhs; ++b) {
            if (lhs % (b * b) == 0) {
                ll c = lhs / (b * b);
                if (a + b + c <= limit) {
                    // Lock mutex to safely add the unique triplet to the shared container
                    std::lock_guard<std::mutex> guard(triplet_mutex);
                    local_triplets.insert(std::make_tuple(a, b, c));  // Add the triplet to the set
                    found_triplet_for_a = true;
                }
            }
        }

        // If no valid triplet was found for the current value of a and the sum exceeds the limit, we stop
        if (!found_triplet_for_a && a > limit) {
            done.store(true);  // Signal threads to stop
            break;
        }

        a += step;  // Skip over other threads' values
    }
}

int main() {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    ll limit = 110000000;  // The sum a + b + c must be less than or equal to this limit
    std::atomic<bool> done(false);

    // Get the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Default to 4 if the system cannot determine it

    // Each thread will have its own local set of triplets
    std::vector<std::unordered_set<std::tuple<ll, ll, ll>, hash_triplet>> thread_triplets(num_threads);

    std::vector<std::thread> threads;

    // Launch multiple threads with different starting points for `a`
    ll step = num_threads * 3;
    for (unsigned int i = 0; i < num_threads; ++i) {
        ll thread_start_a = 2 + 3 * i;  // Each thread starts with a different value of `a`
        threads.emplace_back(find_cardano_triplets, thread_start_a, limit, std::ref(done), std::ref(thread_triplets[i]), step);
    }

    // Join all threads to ensure they complete execution
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Combine results from all threads
    std::unordered_set<std::tuple<ll, ll, ll>, hash_triplet> triplets;
    for (const auto& local_triplets : thread_triplets) {
        triplets.insert(local_triplets.begin(), local_triplets.end());
    }

    // Output the total number of unique triplets found
    std::cout << "Number of unique Cardano Triplets: " << triplets.size() << std::endl;

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Execution time: " << diff.count() << " seconds" << std::endl;

    return 0;
}