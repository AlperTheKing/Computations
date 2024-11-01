#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <bitset>

// Total number of elements
const int NUM_COUNT = 20;
// Numbers from 2 to 21
const std::vector<int> numbers = {2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

// Conflict matrix: conflict[i] as a bitmask indicating numbers that conflict with the i-th number
std::vector<unsigned long> conflict(NUM_COUNT, 0);

// Global maximum subset size
int global_max_size = 0;
// Vector to store all maximum subsets
std::vector<unsigned long> global_subsets;
// Mutex for synchronizing access to global_max_size and global_subsets
std::mutex mtx;

// Worker function for each thread
void worker(unsigned long start, unsigned long end) {
    int local_max_size = 0;
    std::vector<unsigned long> local_subsets;

    for(unsigned long subset = start; subset < end; ++subset){
        // Check for conflicts using the conflict matrix
        bool valid = true;
        for(int i = 0; i < NUM_COUNT; ++i){
            if(subset & (1UL << i)){
                // If any conflicting number is also in the subset, it's invalid
                if(conflict[i] & subset){
                    valid = false;
                    break;
                }
            }
        }

        if(valid){
            // Calculate the size of the subset
            int count = __builtin_popcount(subset);
            if(count > local_max_size){
                local_max_size = count;
                local_subsets.clear();
                local_subsets.push_back(subset);
            }
            else if(count == local_max_size){
                local_subsets.push_back(subset);
            }
        }
    }

    // Update global maximum and collect subsets
    std::lock_guard<std::mutex> lock(mtx);
    if(local_max_size > global_max_size){
        global_max_size = local_max_size;
        global_subsets = local_subsets;
    }
    else if(local_max_size == global_max_size){
        global_subsets.insert(global_subsets.end(), local_subsets.begin(), local_subsets.end());
    }
}

int main(){
    // Build the conflict matrix
    for(int i = 0; i < NUM_COUNT; ++i){
        for(int j = 0; j < NUM_COUNT; ++j){
            if(i == j) continue;
            if(numbers[j] % numbers[i] == 0){
                conflict[i] |= (1UL << j);
            }
        }
    }

    // Determine the number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4; // Default to 4 if unable to detect
    std::cout << "Number of threads used: " << num_threads << std::endl;

    // Total number of subsets
    unsigned long total_subsets = 1UL << NUM_COUNT;

    // Calculate the range of subsets each thread will process
    std::vector<std::thread> threads;
    unsigned long subsets_per_thread = total_subsets / num_threads;
    unsigned long remaining = total_subsets % num_threads;

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    unsigned long current_start = 0;
    for(unsigned int t = 0; t < num_threads; ++t){
        unsigned long current_end = current_start + subsets_per_thread;
        if(t == num_threads -1){
            current_end += remaining; // Last thread handles the remainder
        }
        threads.emplace_back(worker, current_start, current_end);
        current_start = current_end;
    }

    // Wait for all threads to finish
    for(auto &th : threads){
        th.join();
    }

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Display the results
    std::cout << "Maximum subset size: " << global_max_size << std::endl;
    std::cout << "Number of maximum subsets: " << global_subsets.size() << std::endl;
    std::cout << "Maximum subsets:" << std::endl;

    for(const auto& subset : global_subsets){
        std::cout << "{ ";
        for(int i = 0; i < NUM_COUNT; ++i){
            if(subset & (1UL << i)){
                std::cout << numbers[i] << " ";
            }
        }
        std::cout << "}" << std::endl;
    }

    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}