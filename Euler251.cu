#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <set>
#include <tuple>

typedef unsigned long long ull;
typedef __uint128_t ull128; // 128-bit unsigned integer

// CUDA kernel to calculate Cardano Triplets
__global__ void find_cardano_triplets(ull128* triplet_counts, ull128 start_a, ull128 end_a, ull128 max_sum, ull128* primes, int num_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    ull128 a = start_a + idx * 3;
    if (a > end_a) return;

    ull128 N = (1 + a) * (1 + a) * (8 * a - 1);
    if (N % 27 != 0)
        return;
    
    ull128 N_div = N / 27;

    // Factorize N_div using the primes
    ull128 b = 1, c = 1;

    for (int i = 0; i < num_primes; i++) {
        ull128 prime = primes[i];
        if (prime * prime > N_div)
            break;

        ull128 count = 0;
        while (N_div % prime == 0) {
            N_div /= prime;
            count++;
        }

        if (count > 0) {
            ull128 max_k = count / 2;
            for (ull128 k = 0; k <= max_k; ++k) {
                b = b * prime;
                c = c * prime;
                // Only store unique triplets (a, b, c)
                if (a + b + c <= max_sum && b > 0 && c > 0) {
                    atomicAdd(triplet_counts, 1);
                }
            }
        }
    }
}

// Host function to generate a list of primes using the Sieve of Eratosthenes
std::vector<ull128> sieve(ull128 max_n) {
    std::vector<bool> is_prime(static_cast<size_t>(max_n + 1), true);
    std::vector<ull128> primes;
    is_prime[0] = is_prime[1] = false;

    for (ull128 i = 2; i <= max_n; ++i) {
        if (is_prime[static_cast<size_t>(i)]) {
            primes.push_back(i);
            for (ull128 j = i * i; j <= max_n; j += i)
                is_prime[static_cast<size_t>(j)] = false;
        }
    }
    return primes;
}

// Custom output function for __uint128_t
std::ostream& operator<<(std::ostream& dest, __uint128_t value) {
    std::ostream::sentry s(dest);
    if (s) {
        __uint128_t tmp = value;
        char buffer[128];
        char* d = std::end(buffer);
        do {
            --d;
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        dest.write(d, std::end(buffer) - d);
    }
    return dest;
}

int main() {
    ull128 max_sum;
    std::cout << "Enter the maximum value for (a + b + c): ";
    std::cin >> max_sum;

    // Generate primes up to the square root of the largest possible N
    ull128 max_prime = (ull128)sqrt((1 + max_sum) * (1 + max_sum) * (8 * max_sum - 1) / 27);
    std::vector<ull128> host_primes = sieve(max_prime);

    // Allocate memory for primes on the device (GPU)
    ull128* device_primes;
    cudaMalloc(&device_primes, host_primes.size() * sizeof(ull128));
    cudaMemcpy(device_primes, host_primes.data(), host_primes.size() * sizeof(ull128), cudaMemcpyHostToDevice);

    // Allocate memory for counting the triplets
    ull128* device_triplet_count;
    cudaMalloc(&device_triplet_count, sizeof(ull128));
    cudaMemset(device_triplet_count, 0, sizeof(ull128));

    auto start_time = std::chrono::high_resolution_clock::now();

    // Use cudaOccupancyMaxPotentialBlockSize to determine optimal block and grid size
    int block_size;     // The optimal block size
    int min_grid_size;  // The minimum grid size needed to achieve maximum occupancy

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, find_cardano_triplets, 0, 0);

    ull128 total_a_values = max_sum / 3;
    int grid_size = (total_a_values + block_size - 1) / block_size;

    std::cout << "Using block size: " << block_size << ", grid size: " << grid_size << std::endl;

    // Launch the kernel
    find_cardano_triplets<<<grid_size, block_size>>>(device_triplet_count, 2, max_sum, max_sum, device_primes, host_primes.size());

    // Synchronize the device
    cudaDeviceSynchronize();

    // Copy the result back to the host
    ull128 host_triplet_count;
    cudaMemcpy(&host_triplet_count, device_triplet_count, sizeof(ull128), cudaMemcpyDeviceToHost);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Output the total number of unique Cardano triplets found
    std::cout << "Total Cardano Triplets: " << host_triplet_count << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds\n";

    // Free GPU memory
    cudaFree(device_primes);
    cudaFree(device_triplet_count);

    return 0;
}