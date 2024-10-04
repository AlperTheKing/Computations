#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_SIMULATIONS 1000000000000  // 1 trillion simulations
#define THREADS_PER_BLOCK 256  // Number of threads per block

using namespace std;
using namespace std::chrono;

__device__ uint64_t splitmix64(uint64_t *seed) {
    uint64_t result = (*seed += 0x9E3779B97F4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

__device__ void seed_xoshiro512(uint64_t seed, uint64_t *s) {
    for (int i = 0; i < 8; i++) {
        s[i] = splitmix64(&seed);
    }
}

__device__ uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

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

__global__ void simulate_urns(int num_simulations_per_thread, uint64_t *d_red_red_count, uint64_t *d_red_green_count, uint64_t *d_all_red_urn_selected, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

    atomicAdd((unsigned long long*)d_red_red_count, (unsigned long long)local_red_red_count);
    atomicAdd((unsigned long long*)d_red_green_count, (unsigned long long)local_red_green_count);
    atomicAdd((unsigned long long*)d_all_red_urn_selected, (unsigned long long)local_all_red_urn_selected);
}

int main() {
    // Get the number of available GPUs
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 2) {
        cout << "This code requires at least 2 GPUs." << endl;
        return 1;
    }

    // Allocate memory on both devices (GPU 0 and GPU 1)
    uint64_t *d_red_red_count[2], *d_red_green_count[2], *d_all_red_urn_selected[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_red_red_count[i], sizeof(uint64_t));
        cudaMalloc(&d_red_green_count[i], sizeof(uint64_t));
        cudaMalloc(&d_all_red_urn_selected[i], sizeof(uint64_t));

        cudaMemset(d_red_red_count[i], 0, sizeof(uint64_t));
        cudaMemset(d_red_green_count[i], 0, sizeof(uint64_t));
        cudaMemset(d_all_red_urn_selected[i], 0, sizeof(uint64_t));
    }

    // CUDA stream for each device
    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }

    // Determine optimal block size and grid size using cudaOccupancyMaxPotentialBlockSize
    int blockSize = 0;
    int minGridSize = 0;
    cudaSetDevice(0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, simulate_urns, 0, 0);
    int num_blocks = (NUM_SIMULATIONS / 2 / blockSize) + 1;

    // Each device runs half of the simulations
    int simulations_per_thread = NUM_SIMULATIONS / 2 / (num_blocks * blockSize);

    // Start timer
    auto start = high_resolution_clock::now();

    // Launch kernel on both GPUs
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        simulate_urns<<<num_blocks, blockSize, 0, stream[i]>>>(simulations_per_thread, d_red_red_count[i], d_red_green_count[i], d_all_red_urn_selected[i], 123456789 + i);
    }

    // Wait for both GPUs to finish
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(stream[i]);
    }

    // Copy results back to host and combine results from both GPUs
    uint64_t h_red_red_count[2], h_red_green_count[2], h_all_red_urn_selected[2];
    uint64_t total_red_red_count = 0, total_red_green_count = 0, total_all_red_urn_selected = 0;

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMemcpy(&h_red_red_count[i], d_red_red_count[i], sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_red_green_count[i], d_red_green_count[i], sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_all_red_urn_selected[i], d_all_red_urn_selected[i], sizeof(uint64_t), cudaMemcpyDeviceToHost);

        total_red_red_count += h_red_red_count[i];
        total_red_green_count += h_red_green_count[i];
        total_all_red_urn_selected += h_all_red_urn_selected[i];
    }

    // Stop timer
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Output results
    uint64_t total_red_count = total_red_red_count + total_red_green_count;
    double prob_red = (double)total_red_red_count / total_red_count;
    double prob_green = (double)total_red_green_count / total_red_count;

    cout << "Total red balls drawn: " << total_red_count << "\n";
    cout << "Red followed by Red: " << total_red_red_count << "\n";
    cout << "Red followed by Green: " << total_red_green_count << "\n";
    cout << "Urn with all red balls selected: " << total_all_red_urn_selected << " times\n";
    cout << "Probability of next ball being Red: " << prob_red << "\n";
    cout << "Probability of next ball being Green: " << prob_green << "\n";
    cout << "Time taken: " << duration.count() << " seconds\n";

    // Free memory and destroy streams
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(d_red_red_count[i]);
        cudaFree(d_red_green_count[i]);
        cudaFree(d_all_red_urn_selected[i]);
        cudaStreamDestroy(stream[i]);
    }

    return 0;
}