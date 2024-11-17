// Compile with: nvcc -arch=sm_86 -o 4thpowerEquationCuda 4thpowerEquation.cu -O3
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>  // For sleep function

#define MAX_D 1000000            // Reduced for testing
#define MAX_SOLUTIONS 10000000 // Adjust as needed
#define REPORT_INTERVAL 100   // Report every 10 d's

typedef struct {
    unsigned long long int low;
    unsigned long long int high;
} uint128;

// Helper macro for CUDA error checking
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(e));                             \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// Device functions for 128-bit arithmetic
__device__ uint128 add_uint128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < a.low ? 1 : 0);
    return result;
}

__device__ uint128 subtract_uint128(uint128 a, uint128 b) {
    uint128 result;
    result.low = a.low - b.low;
    result.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return result;
}

__device__ int compare_uint128(uint128 a, uint128 b) {
    if (a.high < b.high) return -1;
    if (a.high > b.high) return 1;
    if (a.low < b.low) return -1;
    if (a.low > b.low) return 1;
    return 0;
}

// Device function to compute i^4 as uint128 without using __int128
__device__ uint128 compute_i4(unsigned long long int i) {
    unsigned long long int i2_low = i * i;
    unsigned long long int i2_high = 0; // i <= 100,000, i^2 fits in 64 bits

    // Compute i^3 = i^2 * i
    unsigned long long int i3_low = i2_low * i;
    unsigned long long int carry = 0;
    if (i2_low != 0 && i3_low / i2_low != i) {
        carry = 1;
    }
    unsigned long long int i3_high = i2_high * i + carry;

    // Compute i^4 = i^3 * i
    unsigned long long int i4_low = i3_low * i;
    carry = 0;
    if (i3_low != 0 && i4_low / i3_low != i) {
        carry = 1;
    }
    unsigned long long int i4_high = i3_high * i + carry;

    uint128 result;
    result.low = i4_low;
    result.high = i4_high;
    return result;
}

// Device function to perform linear search for a
__device__ unsigned long long int find_a(uint128 required_a4, unsigned long long int b) {
    for (unsigned long long int a = 1; a <= b; a++) {
        uint128 a4 = compute_i4(a);
        if (compare_uint128(a4, required_a4) == 0) {
            return a;
        }
    }
    return 0; // Not found
}

__global__ void find_solutions(
    unsigned long long int min_d,
    unsigned long long int max_d,
    unsigned long long int *results,
    unsigned long long int *result_count,
    unsigned long long int *progress_counter) // Added progress counter
{
    unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int total_d = max_d - min_d + 1;

    // Each thread processes one value of d
    if (idx >= total_d) return;

    unsigned long long int d = min_d + idx;
    uint128 d4 = compute_i4(d);

    // Iterate over c, b, a with a <= b <= c <= d
    for (unsigned long long int c = d; c >= 1; c--) {
        uint128 c4 = compute_i4(c);
        // Check if c^4 > d^4
        if (compare_uint128(c4, d4) > 0) continue;

        for (unsigned long long int b = c; b >= 1; b--) {
            uint128 b4 = compute_i4(b);
            uint128 cb_sum = add_uint128(c4, b4);

            // Check if cb_sum > d^4
            if (compare_uint128(cb_sum, d4) > 0) continue;

            // Calculate the required a^4 = d^4 - b^4 - c^4
            uint128 required_a4 = subtract_uint128(d4, cb_sum);

            // Find a such that a^4 == required_a4 and a <= b
            unsigned long long int a = find_a(required_a4, b);

            if (a > 0) { // Found a valid a
                // Atomically reserve space for the solution (4 entries: a, b, c, d)
                unsigned long long int res_idx = atomicAdd(result_count, 4ULL);
                if (res_idx + 4 <= MAX_SOLUTIONS * 4) {
                    results[res_idx]     = a;
                    results[res_idx + 1] = b;
                    results[res_idx + 2] = c;
                    results[res_idx + 3] = d;
                }
            }
        }
    }

    // Atomically increment the progress counter
    unsigned long long int current_progress = atomicAdd(progress_counter, 1ULL) + 1ULL;

    // Progress reporting within kernel (for debugging purposes)
    if (current_progress % (REPORT_INTERVAL / 2) == 0) { // Report twice as frequently
        printf("Thread %llu on device %d processed d = %llu\n", idx, blockIdx.x, d);
    }
}

int main()
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices < 2)
    {
        printf("This program requires at least 2 GPUs.\n");
        return 1;
    }

    // Create CUDA streams
    cudaStream_t stream0, stream1;

    // Device 0
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaCheckError();

    // Device 1
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);
    cudaCheckError();

    // Allocate memory for results and result_count on each device
    unsigned long long int *d_results0, *d_result_count0, *d_progress0;
    unsigned long long int *d_results1, *d_result_count1, *d_progress1;

    size_t results_size = MAX_SOLUTIONS * 4 * sizeof(unsigned long long int);

    // Device 0 allocations
    cudaSetDevice(0);
    cudaMalloc((void**)&d_results0, results_size);
    cudaCheckError();
    cudaMalloc((void**)&d_result_count0, sizeof(unsigned long long int));
    cudaCheckError();
    cudaMalloc((void**)&d_progress0, sizeof(unsigned long long int)); // Allocate progress counter
    cudaCheckError();
    cudaMemset(d_result_count0, 0, sizeof(unsigned long long int));
    cudaCheckError();
    cudaMemset(d_progress0, 0, sizeof(unsigned long long int));     // Initialize progress counter
    cudaCheckError();

    // Device 1 allocations
    cudaSetDevice(1);
    cudaMalloc((void**)&d_results1, results_size);
    cudaCheckError();
    cudaMalloc((void**)&d_result_count1, sizeof(unsigned long long int));
    cudaCheckError();
    cudaMalloc((void**)&d_progress1, sizeof(unsigned long long int)); // Allocate progress counter
    cudaCheckError();
    cudaMemset(d_result_count1, 0, sizeof(unsigned long long int));
    cudaCheckError();
    cudaMemset(d_progress1, 0, sizeof(unsigned long long int));     // Initialize progress counter
    cudaCheckError();

    // Divide the range of d between two devices
    unsigned long long int min_d0 = 1ULL;
    unsigned long long int max_d0 = MAX_D / 2ULL;

    unsigned long long int min_d1 = max_d0 + 1ULL;
    unsigned long long int max_d1 = MAX_D;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaCheckError();
    cudaEventCreate(&stop);
    cudaCheckError();
    cudaEventRecord(start, 0);
    cudaCheckError();

    // Launch kernels on both devices
    int threadsPerBlock = 256;
    int blocksPerGrid0 = ((max_d0 - min_d0 + 1ULL) + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid1 = ((max_d1 - min_d1 + 1ULL) + threadsPerBlock - 1) / threadsPerBlock;

    // Device 0 kernel launch
    cudaSetDevice(0);
    find_solutions<<<blocksPerGrid0, threadsPerBlock, 0, stream0>>>(
        min_d0,
        max_d0,
        d_results0,
        d_result_count0,
        d_progress0  // Pass progress counter
    );
    cudaCheckError();

    // Device 1 kernel launch
    cudaSetDevice(1);
    find_solutions<<<blocksPerGrid1, threadsPerBlock, 0, stream1>>>(
        min_d1,
        max_d1,
        d_results1,
        d_result_count1,
        d_progress1  // Pass progress counter
    );
    cudaCheckError();

    // Variables to track progress
    unsigned long long int h_progress0 = 0;
    unsigned long long int h_progress1 = 0;
    unsigned long long int next_report0 = REPORT_INTERVAL;
    unsigned long long int next_report1 = REPORT_INTERVAL;
    bool done0 = false;
    bool done1 = false;
    int timeout = 0;
    const int MAX_TIMEOUT = 100; // e.g., 100 * 100ms = 10 seconds

    // Polling loop for progress reporting
    while (!done0 || !done1)
    {
        // Sleep for a short duration to avoid busy waiting
        usleep(100000); // 100 milliseconds
        timeout++;

        // Check progress for Device 0
        if (!done0) {
            cudaSetDevice(0);
            cudaMemcpy(&h_progress0, d_progress0, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            cudaCheckError();

            while (h_progress0 >= next_report0 && next_report0 <= (max_d0 - min_d0 + 1ULL)) {
                printf("Device 0: Processed %llu / %llu d's.\n", next_report0, max_d0 - min_d0 + 1ULL);
                next_report0 += REPORT_INTERVAL;
            }

            // Check if Device 0 has finished processing
            if (h_progress0 >= (max_d0 - min_d0 + 1ULL)) {
                done0 = true;
                printf("Device 0: Processing complete.\n");
            }
        }

        // Check progress for Device 1
        if (!done1) {
            cudaSetDevice(1);
            cudaMemcpy(&h_progress1, d_progress1, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            cudaCheckError();

            while (h_progress1 >= next_report1 && next_report1 <= (max_d1 - min_d1 + 1ULL)) {
                printf("Device 1: Processed %llu / %llu d's.\n", next_report1, max_d1 - min_d1 + 1ULL);
                next_report1 += REPORT_INTERVAL;
            }

            // Check if Device 1 has finished processing
            if (h_progress1 >= (max_d1 - min_d1 + 1ULL)) {
                done1 = true;
                printf("Device 1: Processing complete.\n");
            }
        }

        // Implement timeout to prevent infinite loop
        if (timeout > MAX_TIMEOUT) {
            printf("Timeout reached. Exiting progress loop.\n");
            break;
        }
    }

    // Wait for kernels to finish
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaCheckError();

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaCheckError();

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaCheckError();
    cudaEventSynchronize(stop);
    cudaCheckError();

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;

    // Copy results back to host
    unsigned long long int h_result_count0 = 0ULL;
    unsigned long long int h_result_count1 = 0ULL;

    cudaSetDevice(0);
    cudaMemcpy(&h_result_count0, d_result_count0, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaSetDevice(1);
    cudaMemcpy(&h_result_count1, d_result_count1, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaCheckError();

    unsigned long long int *h_results0 = NULL;
    unsigned long long int *h_results1 = NULL;

    if (h_result_count0 > 0)
    {
        h_results0 = (unsigned long long int*)malloc(h_result_count0 * sizeof(unsigned long long int));
        if (h_results0 == NULL) {
            printf("Host memory allocation failed for results0.\n");
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(0);
        cudaMemcpy(h_results0, d_results0, h_result_count0 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaCheckError();
    }

    if (h_result_count1 > 0)
    {
        h_results1 = (unsigned long long int*)malloc(h_result_count1 * sizeof(unsigned long long int));
        if (h_results1 == NULL) {
            printf("Host memory allocation failed for results1.\n");
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(1);
        cudaMemcpy(h_results1, d_results1, h_result_count1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaCheckError();
    }

    // Print the results
    printf("Solutions found on Device 0:\n");
    for (unsigned long long int i = 0ULL; i < h_result_count0; i += 4ULL)
    {
        unsigned long long int a = h_results0[i];
        unsigned long long int b = h_results0[i + 1];
        unsigned long long int c = h_results0[i + 2];
        unsigned long long int d = h_results0[i + 3];
        printf("%llu^4 + %llu^4 + %llu^4 = %llu^4\n", a, b, c, d);
    }

    printf("Solutions found on Device 1:\n");
    for (unsigned long long int i = 0ULL; i < h_result_count1; i += 4ULL)
    {
        unsigned long long int a = h_results1[i];
        unsigned long long int b = h_results1[i + 1];
        unsigned long long int c = h_results1[i + 2];
        unsigned long long int d = h_results1[i + 3];
        printf("%llu^4 + %llu^4 + %llu^4 = %llu^4\n", a, b, c, d);
    }

    printf("Total execution time: %f seconds\n", seconds);

    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_results0);
    cudaCheckError();
    cudaFree(d_result_count0);
    cudaCheckError();
    cudaFree(d_progress0); // Free progress counter
    cudaCheckError();
    cudaStreamDestroy(stream0);
    cudaCheckError();
    if (h_results0) free(h_results0);

    cudaSetDevice(1);
    cudaFree(d_results1);
    cudaCheckError();
    cudaFree(d_result_count1);
    cudaCheckError();
    cudaFree(d_progress1); // Free progress counter
    cudaCheckError();
    cudaStreamDestroy(stream1);
    cudaCheckError();
    if (h_results1) free(h_results1);

    cudaEventDestroy(start);
    cudaCheckError();
    cudaEventDestroy(stop);
    cudaCheckError();

    return 0;
}
