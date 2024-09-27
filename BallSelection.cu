#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_SIMULATIONS 1000000000000  // 1 trillion simulations
#define THREADS_PER_BLOCK 256  // Number of threads per block

using namespace std;
using namespace std::chrono;

// Splitmix64 function to seed the xoshiro512**
__device__ uint64_t splitmix64(uint64_t *seed) {
    uint64_t result = (*seed += 0x9E3779B97F4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

// Seed the xoshiro512** generator using splitmix64
__device__ void seed_xoshiro512(uint64_t seed, uint64_t *s) {
    for (int i = 0; i < 8; i++) {
        s[i] = splitmix64(&seed);
    }
}

// Rotate left function
__device__ uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro512** random number generator
__device__ uint64_t xoshiro512_next(uint64_t *s) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 11;
    
    s[2] ^= s[0];
    s[5] ^= s[1];
    s[1] ^= s[2];
    s[7] ^= s[3];
    s[3] ^= s[4];
    s[4] ^= s[5];
    s[0] ^= s[6];
    s[6] ^= s[7];
    s[6] ^= t;
    s[7] = rotl(s[7], 21);
    
    return result;
}

// CUDA kernel to simulate urn problem
__global__ void simulate_urns(int num_simulations_per_thread, uint64_t *d_red_red_count, uint64_t *d_red_green_count, uint64_t *d_all_red_urn_selected, uint64_t seed) {
    // Thread index in the grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets its own state array for xoshiro512**
    uint64_t s[8];
    seed_xoshiro512(seed + idx, s);

    uint64_t local_red_red_count = 0;
    uint64_t local_red_green_count = 0;
    uint64_t local_all_red_urn_selected = 0;

    for (int i = 0; i < num_simulations_per_thread; i++) {
        uint64_t urn_choice = xoshiro512_next(s) % 100;

        if (urn_choice == 99) {
            local_all_red_urn_selected++;
            local_red_red_count++;
        } else {
            uint64_t ball_choice = xoshiro512_next(s) % 99;
            if (ball_choice == 0) {
                local_red_green_count++;
            }
        }
    }

    // Accumulate results in global memory (atomic operations)
    atomicAdd((unsigned long long*)d_red_red_count, (unsigned long long)local_red_red_count);
    atomicAdd((unsigned long long*)d_red_green_count, (unsigned long long)local_red_green_count);
    atomicAdd((unsigned long long*)d_all_red_urn_selected, (unsigned long long)local_all_red_urn_selected);
}

int main() {
    // Get number of threads supported by GPU
    int num_threads = THREADS_PER_BLOCK * 256;  // Adjust for your GPU's capacity

    // Allocate memory on the device
    uint64_t *d_red_red_count, *d_red_green_count, *d_all_red_urn_selected;
    cudaMalloc(&d_red_red_count, sizeof(uint64_t));
    cudaMalloc(&d_red_green_count, sizeof(uint64_t));
    cudaMalloc(&d_all_red_urn_selected, sizeof(uint64_t));

    // Initialize device memory
    cudaMemset(d_red_red_count, 0, sizeof(uint64_t));
    cudaMemset(d_red_green_count, 0, sizeof(uint64_t));
    cudaMemset(d_all_red_urn_selected, 0, sizeof(uint64_t));

    // Number of simulations per thread
    int simulations_per_thread = NUM_SIMULATIONS / num_threads;

    // Start timer
    auto start = high_resolution_clock::now();

    // Launch the CUDA kernel
    simulate_urns<<<num_threads / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(simulations_per_thread, d_red_red_count, d_red_green_count, d_all_red_urn_selected, 123456789);

    // Copy results back to host
    uint64_t h_red_red_count, h_red_green_count, h_all_red_urn_selected;
    cudaMemcpy(&h_red_red_count, d_red_red_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_red_green_count, d_red_green_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_all_red_urn_selected, d_all_red_urn_selected, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Stop timer
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Output results
    uint64_t total_red_count = h_red_red_count + h_red_green_count;
    double prob_red = (double)h_red_red_count / total_red_count;
    double prob_green = (double)h_red_green_count / total_red_count;

    cout << "Total red balls drawn: " << total_red_count << "\n";
    cout << "Red followed by Red: " << h_red_red_count << "\n";
    cout << "Red followed by Green: " << h_red_green_count << "\n";
    cout << "Urn with all red balls selected: " << h_all_red_urn_selected << " times\n";
    cout << "Probability of next ball being Red: " << prob_red << "\n";
    cout << "Probability of next ball being Green: " << prob_green << "\n";
    cout << "Time taken: " << duration.count() << " seconds\n";

    // Free memory
    cudaFree(d_red_red_count);
    cudaFree(d_red_green_count);
    cudaFree(d_all_red_urn_selected);

    return 0;
}