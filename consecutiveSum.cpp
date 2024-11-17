#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>

// ThreadPool class for managing a pool of worker threads
class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Enqueue a task into the thread pool
    void enqueue(std::function<void()> task);

    // Wait for all tasks to complete
    void wait_for_completion();

private:
    // Worker function for each thread
    void worker();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    std::mutex completion_mutex;
    std::condition_variable completion_condition;
    size_t tasks_in_progress;
};

ThreadPool::ThreadPool(size_t num_threads) : stop(false), tasks_in_progress(0) {
    for (size_t i = 0; i < num_threads; ++i)
        workers.emplace_back(&ThreadPool::worker, this);
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(task);
        ++tasks_in_progress;
    }
    condition.notify_one();
}

void ThreadPool::wait_for_completion() {
    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_condition.wait(lock, [this]() { return tasks_in_progress == 0 && tasks.empty(); });
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            // Acquire task
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this]() { return stop || !tasks.empty(); });
            if (stop && tasks.empty())
                return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        // Execute task
        task();
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            --tasks_in_progress;
            if (tasks_in_progress == 0 && tasks.empty()) {
                completion_condition.notify_all();
            }
        }
    }
}

// Function to generate all prime numbers up to 'limit' using Sieve of Eratosthenes
std::vector<int64_t> generatePrimes(int64_t limit) {
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    for(int64_t p = 2; p*p <= limit; ++p){
        if(is_prime[p]){
            for(int64_t multiple = p*p; multiple <= limit; multiple += p){
                is_prime[multiple] = false;
            }
        }
    }
    std::vector<int64_t> primes;
    for(int64_t p = 2; p <= limit; ++p){
        if(is_prime[p]){
            primes.push_back(p);
        }
    }
    return primes;
}

// Function to calculate the number of odd divisors of 'N'
int countOddDivisors(int64_t N, const std::vector<int64_t>& primes) {
    // Remove all factors of 2
    while(N % 2 == 0){
        N /= 2;
    }
    if(N == 1){
        return 1; // Only one odd divisor (1)
    }
    int count = 1;
    for(auto prime : primes){
        if(prime * prime > N){
            break;
        }
        if(N % prime == 0){
            int exponent = 0;
            while(N % prime == 0){
                N /= prime;
                exponent++;
            }
            count *= (exponent + 1);
        }
    }
    if(N > 1){
        // N is a prime number itself
        count *= 2;
    }
    return count;
}

int main(){
    // Define the ranges: 10^1, 10^2, ..., 10^10
    std::vector<int64_t> ranges;
    for(int i = 1; i <= 10; ++i){
        ranges.push_back(static_cast<int64_t>(pow(10, i)));
    }

    // Precompute primes up to sqrt(10^10) which is 10^5
    // Adjust if necessary based on the maximum range
    int64_t prime_limit = static_cast<int64_t>(sqrt(1e10)) + 1;
    std::cout << "Generating primes up to " << prime_limit << "...\n";
    auto prime_start = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> primes = generatePrimes(prime_limit);
    auto prime_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prime_duration = prime_end - prime_start;
    std::cout << "Prime generation completed in " << prime_duration.count() << " seconds.\n\n";

    // Determine the number of available hardware threads
    size_t num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0){
        num_threads = 4; // Fallback to 4 threads if hardware_concurrency cannot determine
    }
    std::cout << "Initializing ThreadPool with " << num_threads << " threads.\n\n";

    // Initialize ThreadPool
    ThreadPool pool(num_threads);

    // Iterate through each range
    int64_t previous_limit = 0;
    for(auto limit : ranges){
        std::cout << "Processing range: 1 to " << limit << "\n";

        // Start time measurement for this range
        auto start_time = std::chrono::high_resolution_clock::now();

        // Variables to track maximum count and corresponding numbers
        std::mutex result_mutex;
        int global_max_count = 0;
        std::vector<int64_t> global_numbers_with_max_count;

        // Define chunk size (number of N's per task)
        const int64_t chunk_size = 100000; // Adjust based on performance

        // Function to process a chunk of numbers
        auto process_chunk = [&](int64_t start_N, int64_t end_N) {
            int local_max = 0;
            std::vector<int64_t> local_numbers;

            for(int64_t N = start_N; N <= end_N; ++N){
                int odd_divisors = countOddDivisors(N, primes);
                if(odd_divisors > local_max){
                    local_max = odd_divisors;
                    local_numbers.clear();
                    local_numbers.push_back(N);
                }
                else if(odd_divisors == local_max){
                    local_numbers.push_back(N);
                }
            }

            // Update global maximum and numbers with thread safety
            std::lock_guard<std::mutex> lock(result_mutex);
            if(local_max > global_max_count){
                global_max_count = local_max;
                global_numbers_with_max_count = local_numbers;
            }
            else if(local_max == global_max_count){
                global_numbers_with_max_count.insert(global_numbers_with_max_count.end(),
                                                     local_numbers.begin(), local_numbers.end());
            }
        };

        // Enqueue tasks in the thread pool
        for(int64_t chunk_start = previous_limit + 1; chunk_start <= limit; chunk_start += chunk_size){
            int64_t chunk_end = std::min(chunk_start + chunk_size -1, limit);
            pool.enqueue([=, &process_chunk]() {
                process_chunk(chunk_start, chunk_end);
            });
        }

        // Wait for all tasks to complete
        pool.wait_for_completion();

        // End time measurement for this range
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        // Display the results for this range
        std::cout << "Numbers with the maximum number of consecutive sum representations up to " << limit << ":\n";
        for(auto num : global_numbers_with_max_count){
            std::cout << "Number: " << num << " - Number of representations: " << global_max_count << "\n";
        }
        std::cout << "Time taken for this range: " << duration.count() << " seconds.\n\n";

        // Update previous_limit for the next range
        previous_limit = limit;
    }

    return 0;
}