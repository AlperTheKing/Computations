#include <bits/stdc++.h>
#include <gmpxx.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <queue>
#include <atomic>
#include <iostream>
#include <memory>

// Function to generate a string with '1' followed by 'n' zeros
std::string generate_power_of_ten(int exponent) {
    if (exponent <= 0) return "1";
    return "1" + std::string(exponent, '0');
}

// Function to generate ranges up to 10^max_exponent
std::vector<mpz_class> generate_ranges(int max_exponent) {
    std::vector<mpz_class> ranges;
    for(int exp = 1; exp <= max_exponent; ++exp){
        std::string range_str = generate_power_of_ten(exp);
        ranges.emplace_back(range_str);
    }
    return ranges;
}

// Work-stealing ThreadPool class
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Submit a task to the thread pool
    void enqueue(std::function<void()> task);

    // Wait for all tasks to complete
    void wait_until_done();

private:
    std::vector<std::thread> workers;
    std::vector<std::deque<std::function<void()>>> task_queues; // One queue per thread
    std::mutex global_mutex; // Mutex for task queue management
    std::condition_variable condition;
    std::atomic<bool> stop;

    void worker_thread(int thread_id);
};

// Constructor: Launches worker threads
ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    task_queues.resize(num_threads);
    for(size_t i = 0; i < num_threads; ++i){
        workers.emplace_back(&ThreadPool::worker_thread, this, i);
    }
}

// Destructor: Joins all threads
ThreadPool::~ThreadPool(){
    {
        std::unique_lock<std::mutex> lock(global_mutex);
        stop.store(true);
        condition.notify_all();
    }
    for(std::thread &worker: workers)
        worker.join();
}

// Enqueue a new task
void ThreadPool::enqueue(std::function<void()> task){
    {
        std::unique_lock<std::mutex> lock(global_mutex);
        // Push task to the queue of thread 0
        task_queues[0].push_back(task);
        condition.notify_all();
    }
}

// Wait until all tasks are completed
void ThreadPool::wait_until_done(){
    while(true){
        {
            std::unique_lock<std::mutex> lock(global_mutex);
            bool all_empty = true;
            for(auto& queue : task_queues){
                if(!queue.empty()){
                    all_empty = false;
                    break;
                }
            }
            if(all_empty) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Worker thread function
void ThreadPool::worker_thread(int thread_id){
    while(true){
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(global_mutex);
            condition.wait(lock, [this, thread_id]{
                return stop.load() || !task_queues[thread_id].empty();
            });

            if(stop.load()) return;

            if(!task_queues[thread_id].empty()){
                task = task_queues[thread_id].front();
                task_queues[thread_id].pop_front();
            } else {
                // Work stealing
                bool found = false;
                for(size_t i = 0; i < task_queues.size(); ++i){
                    if(i == thread_id) continue;
                    if(!task_queues[i].empty()){
                        task = task_queues[i].back();
                        task_queues[i].pop_back();
                        found = true;
                        break;
                    }
                }
                if(!found){
                    continue;
                }
            }
        }
        // Execute task outside the lock
        task();
    }
}

// Structure to hold range results
struct RangeResult {
    mpz_class max_num;
    unsigned long long max_divisors;
    std::vector<mpz_class> numbers;
    std::mutex range_mutex; // Mutex for each range to ensure thread safety

    // Constructor to initialize max_num
    RangeResult(const mpz_class& num) : max_num(num), max_divisors(0), numbers() {}
};

// Function to generate all primes up to a given limit using the Sieve of Eratosthenes
std::vector<int> sieve_primes(int limit){
    std::vector<bool> is_prime(limit +1, true);
    is_prime[0] = is_prime[1] = false;
    for(int p=2; p*p <= limit; ++p){
        if(is_prime[p]){
            for(int multiple=p*p; multiple <= limit; multiple += p){
                is_prime[multiple] = false;
            }
        }
    }
    std::vector<int> primes;
    for(int p=2; p<=limit; ++p){
        if(is_prime[p]){
            primes.push_back(p);
        }
    }
    return primes;
}

// Recursive function to generate numbers and update range results on the fly
void generateNumbers(mpz_class current, int index, unsigned long long current_divisors, 
                    const mpz_class& max_limit, 
                    std::vector<std::shared_ptr<RangeResult>> &range_results, 
                    const std::vector<int>& primes,
                    ThreadPool& pool, int max_exponents_per_prime[]){
    // Check if current number is within the limit
    if (current > max_limit) return;

    // Iterate through each range
    for(auto &range : range_results){
        if(current <= range->max_num){
            std::lock_guard<std::mutex> lock(range->range_mutex);
            if(current_divisors > range->max_divisors){
                range->max_divisors = current_divisors;
                range->numbers.clear();
                range->numbers.push_back(current);
            }
            else if(current_divisors == range->max_divisors){
                // Avoid duplicates
                if(std::find(range->numbers.begin(), range->numbers.end(), current) == range->numbers.end()){
                    range->numbers.push_back(current);
                }
            }
        }
    }

    if (index >= primes.size()) return;

    // Iterate through possible exponents for the current prime
    for(int exp = 1; exp <= max_exponents_per_prime[index]; ++exp){
        mpz_class next = current;
        bool overflow = false;

        // Compute next = current * (prime[index] ^ exp)
        mpz_class prime_power = 1;
        for(int e=0; e < exp; ++e){
            prime_power *= primes[index];
            if(prime_power > max_limit){
                overflow = true;
                break;
            }
        }
        if(overflow) break;

        next *= prime_power;
        if(next > max_limit) break;

        // Update the number of odd divisors
        unsigned long long new_divisors = current_divisors * (exp +1);

        // Create a copy of the state for the recursive call
        auto task = [next, index_plus_one = index +1, new_divisors, &max_limit, &range_results, &primes, &pool, max_exponents_per_prime](){
            generateNumbers(next, index_plus_one, new_divisors, max_limit, range_results, primes, pool, max_exponents_per_prime);
        };

        // Enqueue the new task
        pool.enqueue(task);
    }
}

int main(){
    // 1. Adjusted Sieve Limit for 10^50
    // The sieve_limit is increased to include more primes
    int sieve_limit = 541; // The 100th prime
    std::vector<int> primes = sieve_primes(sieve_limit);

    // Remove the prime 2 since we're only interested in odd divisors
    primes.erase(std::remove(primes.begin(), primes.end(), 2), primes.end());

    // Verification Step: Ensure that 2 has been removed
    if (std::find(primes.begin(), primes.end(), 2) != primes.end()) {
        std::cerr << "Error: Prime number 2 was not successfully removed from the primes list.\n";
        return 1;
    }

    // 2. Adjusted Maximum Exponent for 10^50
    int max_exponent = 50; // Adjusted for 10^50

    // Generate the ranges up to 10^50
    std::vector<mpz_class> ranges = generate_ranges(max_exponent);

    // Initialize range results with unsigned long long for max_divisors
    std::vector<std::shared_ptr<RangeResult>> range_results;
    range_results.reserve(ranges.size());
    for(const auto& r : ranges){
        range_results.emplace_back(std::make_shared<RangeResult>(r));
    }

    // 3. Initialize ThreadPool
    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4; // Fallback to 4 threads if unable to detect

    ThreadPool pool(num_threads);

    // Correctly Generate max_limit for 10^50
    int desired_exponent = 50;
    std::string max_limit_str = generate_power_of_ten(desired_exponent);
    mpz_class max_limit(max_limit_str); // 10^50

    // Verification of max_limit
    mpz_class expected_max_limit = 1;
    for(int i = 0; i < desired_exponent; ++i){
        expected_max_limit *= 10;
    }
    if(max_limit != expected_max_limit){
        std::cerr << "Error: max_limit is incorrectly generated.\n";
        return 1;
    }

    // Start timing the generation process
    auto start = std::chrono::high_resolution_clock::now();

    // Determine the maximum possible exponent for each prime
    int num_primes = primes.size();
    int* max_exponents_per_prime = new int[num_primes];

    for(int i = 0; i < num_primes; ++i){
        int max_exp = 0;
        mpz_class temp = 1;
        while(temp <= max_limit){
            temp *= primes[i];
            if(temp > max_limit) break;
            max_exp++;
        }
        max_exponents_per_prime[i] = max_exp;
    }

    // Enqueue the initial task to start the recursive generation
    pool.enqueue([&range_results, &primes, &max_limit, &pool, max_exponents_per_prime](){
        generateNumbers(1, 0, 1, max_limit, range_results, primes, pool, max_exponents_per_prime);
    });

    // Wait for all tasks to complete
    pool.wait_until_done();

    // End generation timing
    auto gen_end = std::chrono::high_resolution_clock::now();

    // 4. Output the results for each range
    for(const auto& range : range_results){
        std::cout << "Range: 1-" << range->max_num.get_str() << std::endl;
        std::cout << "Maximum number of different representations (odd divisors): " << range->max_divisors << std::endl;
        std::cout << "Numbers with maximum divisors: ";
        for(auto &num : range->numbers){
            std::cout << num.get_str() << " ";
        }
        std::cout << "\n\n";
    }

    // End overall timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate durations in milliseconds
    auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - start).count();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Generation Time: " << gen_duration << " ms" << std::endl;
    std::cout << "Total Time: " << total_duration << " ms" << std::endl;

    delete[] max_exponents_per_prime;

    return 0;
}
