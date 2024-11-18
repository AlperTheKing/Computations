// consecutiveSumCuda.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <iomanip> // For std::setw

// Macro for CUDA error checking
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

// Function to generate primes up to 'limit' using Sieve of Eratosthenes
std::vector<long long> generatePrimes(long long limit) {
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    for(long long p = 2; p*p <= limit; ++p){
        if(is_prime[p]){
            for(long long multiple = p*p; multiple <= limit; multiple += p){
                is_prime[multiple] = false;
            }
        }
    }
    std::vector<long long> primes;
    for(long long p = 2; p <= limit; ++p){
        if(is_prime[p]){
            primes.push_back(p);
        }
    }
    return primes;
}

// CUDA kernel to count odd divisors and find the local maximum and corresponding numbers
__global__ void countOddDivisorsMaxKernel(
    long long start_N,
    long long end_N,
    const long long* primes,
    int num_primes,
    long long* local_max_count,
    long long* local_max_num
) {
    extern __shared__ long long shared_data[]; // Shared memory for local maximum and number
    // Initialize shared memory
    if (threadIdx.x == 0) {
        shared_data[0] = 0; // local_max_count
        shared_data[1] = 0; // local_max_num
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long N = start_N + idx;
    if(N > end_N) return;

    // Count odd divisors
    int count = 1;
    long long temp = N;

    // Remove factors of 2
    while(temp % 2 == 0){
        temp /= 2;
    }

    // Factorize and count divisors
    for(int i = 0; i < num_primes && primes[i]*primes[i] <= temp; ++i){
        if(temp % primes[i] == 0){
            int exponent = 0;
            while(temp % primes[i] == 0){
                temp /= primes[i];
                exponent++;
            }
            count *= (exponent + 1);
        }
    }

    if(temp > 1){
        // N is a prime number itself
        count *= 2;
    }

    // Atomically update the shared local maximum
    atomicMax(&shared_data[0], (long long)count);

    // If this thread's count equals the shared maximum, store the number
    if(count == shared_data[0]){
        atomicMax(&shared_data[1], N); // Keep the largest N with the max count
    }

    __syncthreads();

    // Write the shared local maximum and number to global memory
    if(threadIdx.x == 0){
        atomicMax(local_max_count, shared_data[0]);
        atomicMax(local_max_num, shared_data[1]);
    }
}

// Function to handle GPU processing in a separate thread
void processGPU(int gpu_id, long long start_N, long long end_N, const long long* d_primes, int num_primes, 
               long long& global_max_count, long long& global_max_num, long long max_batch_size, std::mutex& mtx){
    cudaSetDevice(gpu_id);
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));

    long long local_max_count = 0;
    long long local_max_num = 0;

    // Calculate total batches
    long long total_batches = (end_N - start_N + max_batch_size) / max_batch_size;
    long long batch_counter = 0;

    // Process the subrange in batches
    for(long long batch_start = start_N; batch_start <= end_N; batch_start += max_batch_size){
        long long batch_end = std::min(batch_start + max_batch_size - 1, end_N);
        long long batch_size = batch_end - batch_start + 1;
        batch_counter++;

        // Allocate memory for local max count and number
        long long* d_local_max_count;
        long long* d_local_max_num;
        cudaCheckError(cudaMalloc(&d_local_max_count, sizeof(long long)));
        cudaCheckError(cudaMalloc(&d_local_max_num, sizeof(long long)));
        cudaCheckError(cudaMemsetAsync(d_local_max_count, 0, sizeof(long long), stream));
        cudaCheckError(cudaMemsetAsync(d_local_max_num, 0, sizeof(long long), stream));

        // Determine optimal block size and grid size using occupancy calculator
        int minGridSize;
        int blockSize;
        cudaCheckError(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            countOddDivisorsMaxKernel,
            2 * sizeof(long long), // Shared memory size
            0
        ));

        // Calculate grid size based on the number of elements and optimal block size
        int gridSize = (batch_size + blockSize - 1) / blockSize;

        // Launch the kernel with optimized block and grid sizes
        countOddDivisorsMaxKernel<<<gridSize, blockSize, 2 * sizeof(long long), stream>>>(
            batch_start,
            batch_end,
            d_primes,
            num_primes,
            d_local_max_count,
            d_local_max_num
        );
        cudaCheckError(cudaGetLastError());

        // Allocate host memory for local max count and number
        long long h_local_max_count = 0;
        long long h_local_max_num = 0;

        // Copy the local maximum count and number back to host asynchronously
        cudaCheckError(cudaMemcpyAsync(&h_local_max_count, d_local_max_count, sizeof(long long), cudaMemcpyDeviceToHost, stream));
        cudaCheckError(cudaMemcpyAsync(&h_local_max_num, d_local_max_num, sizeof(long long), cudaMemcpyDeviceToHost, stream));

        // Synchronize the stream to ensure all operations are complete before freeing memory
        cudaCheckError(cudaStreamSynchronize(stream));

        // Free device memory after ensuring all operations are complete
        cudaCheckError(cudaFree(d_local_max_count));
        cudaCheckError(cudaFree(d_local_max_num));

        // Update the local maximum count and corresponding number
        if(h_local_max_count > local_max_count){
            local_max_count = h_local_max_count;
            local_max_num = h_local_max_num;
        }
        else if(h_local_max_count == local_max_count){
            // If multiple numbers have the same maximum count, keep the largest number
            local_max_num = std::max(local_max_num, h_local_max_num);
        }

        // Log progress
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "GPU " << gpu_id << " processed batch " << batch_counter << "/" << total_batches 
                      << " (Numbers " << batch_start << " to " << batch_end << ")\n";
        }
    }

    // Update the global maximum count and corresponding number using mutex for thread safety
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(local_max_count > global_max_count){
            global_max_count = local_max_count;
            global_max_num = local_max_num;
        }
        else if(local_max_count == global_max_count){
            global_max_num = std::max(global_max_num, local_max_num);
        }
    }

    // Destroy the stream
    cudaCheckError(cudaStreamDestroy(stream));
}

int main(){
    // Define the ranges: 10^1, 10^2, ..., 10^10
    std::vector<long long> ranges;
    for(int i = 1; i <= 10; ++i){
        ranges.push_back(static_cast<long long>(pow(10, i)));
    }

    // Precompute primes up to sqrt(10^10) which is 10^5 + 1
    long long prime_limit = static_cast<long long>(sqrt(1e10)) + 1;
    std::cout << "Generating primes up to " << prime_limit << "...\n";
    auto prime_start = std::chrono::high_resolution_clock::now();
    std::vector<long long> primes = generatePrimes(prime_limit);
    auto prime_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prime_duration = prime_end - prime_start;
    std::cout << "Prime generation completed in " << prime_duration.count() << " seconds.\n\n";

    // Determine the number of GPUs
    int device_count = 0;
    cudaCheckError(cudaGetDeviceCount(&device_count));
    if(device_count < 2){
        std::cerr << "This program requires at least 2 CUDA-capable GPUs.\n";
        return 1;
    }
    std::cout << "Number of CUDA-capable GPUs available: " << device_count << "\n\n";

    // Allocate and copy primes to both GPUs
    int num_primes = primes.size();
    long long* d_primes[2];
    for(int gpu = 0; gpu < 2; ++gpu){
        cudaSetDevice(gpu);
        cudaCheckError(cudaMalloc(&d_primes[gpu], num_primes * sizeof(long long)));
        cudaCheckError(cudaMemcpy(d_primes[gpu], primes.data(), num_primes * sizeof(long long), cudaMemcpyHostToDevice));
    }

    // Define maximum batch size based on GPU memory (adjust as needed)
    long long max_batch_size = 10000000; // 1e7

    // Mutex for thread-safe global maximum updates
    std::mutex mtx;

    // Iterate through each range
    for(auto limit : ranges){
        std::cout << "Processing range: 1 to " << limit << "\n";

        // Start time measurement
        auto start_time = std::chrono::high_resolution_clock::now();

        // Split the range into 2/3 and 1/3 for GPU0 and GPU1 respectively
        long long two_thirds = (limit * 2) / 3;

        // Define subranges for each GPU
        struct SubRange {
            int gpu_id;
            long long start_N;
            long long end_N;
        };

        SubRange subranges[2] = {
            {0, 1, two_thirds},
            {1, two_thirds + 1, limit}
        };

        // Log subrange assignments
        for(int gpu = 0; gpu < 2; ++gpu){
            std::cout << "GPU " << subranges[gpu].gpu_id 
                      << " processing numbers " << subranges[gpu].start_N 
                      << " to " << subranges[gpu].end_N << "\n";
        }

        // Variables to store maximum counts from each GPU
        long long global_max_count = 0;
        long long global_max_num = 0;

        // Launch threads for each GPU
        std::thread threads[2];
        for(int gpu = 0; gpu < 2; ++gpu){
            threads[gpu] = std::thread(processGPU, 
                subranges[gpu].gpu_id, 
                subranges[gpu].start_N, 
                subranges[gpu].end_N, 
                d_primes[gpu], 
                num_primes, 
                std::ref(global_max_count), 
                std::ref(global_max_num), 
                max_batch_size, 
                std::ref(mtx));
        }

        // Wait for all threads to finish
        for(int gpu = 0; gpu < 2; ++gpu){
            if(threads[gpu].joinable()){
                threads[gpu].join();
            }
        }

        // Display the maximum count and corresponding number
        std::cout << "Maximum number of odd divisors in this range: " << global_max_count << "\n";
        std::cout << "Number with the maximum number of odd divisors: " << global_max_num << "\n";

        // End time measurement
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::cout << "Time taken for this range: " << duration.count() << " seconds.\n\n";
    }

    // Free device memory for primes
    for(int gpu = 0; gpu < 2; ++gpu){
        cudaSetDevice(gpu);
        cudaCheckError(cudaFree(d_primes[gpu]));
    }

    return 0;
}
