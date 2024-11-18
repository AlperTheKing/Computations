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

// ThreadPool class to manage a pool of worker threads
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
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

    std::mutex done_mutex;
    std::condition_variable done_condition;
    std::atomic<int> active_tasks;
};

// Constructor: Launches worker threads
ThreadPool::ThreadPool(size_t num_threads) : stop(false), active_tasks(0) {
    for(size_t i = 0; i < num_threads; ++i){
        workers.emplace_back([this]() {
            while(true){
                std::function<void()> task;

                // Acquire task
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{
                        return this->stop.load() || !this->tasks.empty();
                    });

                    if(this->stop.load() && this->tasks.empty())
                        return;

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                // Execute task
                task();
                active_tasks--;

                // Notify if all tasks are done
                if(active_tasks.load() == 0 && tasks.empty()){
                    std::unique_lock<std::mutex> lock(this->done_mutex);
                    this->done_condition.notify_all();
                }
            }
        });
    }
}

// Destructor: Joins all threads
ThreadPool::~ThreadPool(){
    stop.store(true);
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

// Enqueue a new task
void ThreadPool::enqueue(std::function<void()> task){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(task);
    }
    active_tasks++;
    condition.notify_one();
}

// Wait until all tasks are completed
void ThreadPool::wait_until_done(){
    std::unique_lock<std::mutex> lock(done_mutex);
    done_condition.wait(lock, [this]{
        return this->tasks.empty() && this->active_tasks.load() == 0;
    });
}

// Structure to hold range results
struct RangeResult {
    mpz_class max_num;
    unsigned long long max_divisors; // Changed to unsigned long long
    std::vector<mpz_class> numbers;
};

// Function to generate all primes up to a given limit using the Sieve of Eratosthenes
std::vector<int> sieve_primes(int limit){
    std::vector<char> is_prime(limit +1, 1); // 1 = prime, 0 = not prime
    is_prime[0] = is_prime[1] = 0;
    for(int p=2; p*p <= limit; ++p){
        if(is_prime[p]){
            for(int multiple=p*p; multiple <= limit; multiple += p){
                is_prime[multiple] = 0;
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
void generateNumbers(mpz_class current, int index, int last_exp, unsigned long long current_divisors, 
                    const mpz_class& max_limit, 
                    std::vector<RangeResult> &range_results, 
                    const std::vector<int>& primes, 
                    std::mutex &result_mutex){
    // Check if current number is within the limit
    if (current > max_limit) return;

    // Update range results
    {
        std::lock_guard<std::mutex> lock(result_mutex);
        for(auto &range : range_results){
            if(current <= range.max_num){
                if(current_divisors > range.max_divisors){
                    range.max_divisors = current_divisors;
                    range.numbers.clear();
                    range.numbers.push_back(current);
                    // Optional: Logging for debugging
                    // std::cout << "New max_divisors for range up to " << range.max_num.get_str() 
                    //           << ": " << range.max_divisors << " by " << current.get_str() << std::endl;
                }
                else if(current_divisors == range.max_divisors){
                    range.numbers.push_back(current);
                    // Optional: Logging for debugging
                    // std::cout << "Another number with max_divisors for range up to " << range.max_num.get_str() 
                    //           << ": " << current.get_str() << std::endl;
                }
            }
        }
    }

    // If all primes have been processed, return
    if (index >= primes.size()) return;

    // Iterate through possible exponents for the current prime
    for(int exp = 1; exp <= last_exp; ++exp){
        // Check if multiplication would exceed the limit
        mpz_class next = current;
        bool overflow = false;
        for(int e=0; e < exp; ++e){
            mpz_class temp = next * primes[index];
            if(temp > max_limit){
                overflow = true;
                break;
            }
            next = temp;
        }
        if(overflow) break;

        // Update the number of odd divisors
        unsigned long long new_divisors = current_divisors * (exp +1);

        // Recursive call with next, index +1, last_exp=exp
        generateNumbers(next, index +1, exp, new_divisors, max_limit, range_results, primes, result_mutex);
    }
}

int main(){
    // Define the sieve limit (reduced for efficiency)
    int sieve_limit = 15485863; // 1,000,000th prime is 15,485,863
    std::cout << "Generating sieve primes up to " << sieve_limit << "..." << std::endl;
    std::vector<int> primes = sieve_primes(sieve_limit);
    std::cout << "Sieve generated with " << primes.size() << " primes.\n" << std::endl;

    // Remove the prime 2 since we're only interested in odd divisors
    primes.erase(std::remove(primes.begin(), primes.end(), 2), primes.end());

    // **Verification Step: Ensure that 2 has been removed**
    if (std::find(primes.begin(), primes.end(), 2) != primes.end()) {
        std::cerr << "Error: Prime number 2 was not successfully removed from the primes list.\n";
        return 1;
    }

    // Define the maximum exponent for the ranges
    int max_exponent = 100; // Adjusted for practicality during testing

    // Generate the ranges up to 10^max_exponent
    std::vector<mpz_class> ranges = generate_ranges(max_exponent);

    // Initialize range results with unsigned long long for max_divisors
    std::vector<RangeResult> range_results;
    for(const auto& r : ranges){
        range_results.emplace_back(RangeResult{r, 0, {}});
    }

    std::mutex result_mutex;

    // Start timing the generation process
    auto start = std::chrono::high_resolution_clock::now();

    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4; // Fallback to 4 threads if unable to detect

    ThreadPool pool(num_threads);

    // Initialize the generation with different exponents of the first prime to distribute tasks
    if(primes.empty()){
        std::cerr << "Error: No primes generated. Exiting.\n";
        return 1;
    }

    int first_prime = primes[0];

    // **Correctly Generate max_limit for 10^80**
    int desired_exponent = 100; // Adjusted for practicality during testing
    std::string max_limit_str = generate_power_of_ten(desired_exponent);
    mpz_class max_limit(max_limit_str); // 10^20

    // Verification of max_limit
    mpz_class expected_max_limit = 1;
    for(int i = 0; i < desired_exponent; ++i){
        expected_max_limit *= 10;
    }
    if(max_limit != expected_max_limit){
        std::cerr << "Error: max_limit is incorrectly generated.\n";
        return 1;
    }

    // Determine the maximum possible exponent for the first prime
    int max_exp_first_prime = 0;
    mpz_class temp = first_prime;
    while(temp <= max_limit){
        temp *= first_prime;
        max_exp_first_prime++;
    }

    // Enqueue tasks for different exponents of the first prime
    for(int exp = 0; exp <= max_exp_first_prime; ++exp){
        pool.enqueue([=, &range_results, &result_mutex, &primes, &max_limit](){
            mpz_class current = 1;
            unsigned long long current_divisors = 1;

            if(exp > 0){
                current = first_prime;
                current_divisors *= (exp + 1);
                for(int e = 1; e < exp; ++e){
                    current *= first_prime;
                }
                if(current > max_limit){
                    // Overflow, skip this task
                    return;
                }
            }

            // Recursively generate numbers and update range results on the fly
            generateNumbers(current, 1, exp, current_divisors, max_limit, range_results, primes, result_mutex);
        });
    }

    // Wait for all tasks to complete
    pool.wait_until_done();

    // End generation timing
    auto gen_end = std::chrono::high_resolution_clock::now();

    // Output the results for each range
    for(const auto& range : range_results){
        std::cout << "Range: 1-" << range.max_num.get_str() << std::endl;
        std::cout << "Maximum number of different representations: " << range.max_divisors << std::endl;
        std::cout << "Numbers: ";
        for(auto &num : range.numbers){
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

    return 0;
}
