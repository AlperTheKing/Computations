// fourSquareRepresentationCuda_parity_simultaneous.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <inttypes.h>
#include <fstream>
#include <thread>

// Struct to hold the representation of a number as the sum of four squares
struct Representation {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t d;
};

// Device function to find the representation of a number as the sum of up to four squares
__device__ Representation findRepresentationDevice(int64_t n) {
    Representation rep = {0, 0, 0, 0};

    // Check for one square
    int64_t a = (int64_t)(sqrt((double)n));
    if (a * a == n) {
        rep.a = a;
        return rep;
    }

    // Check for two squares
    for (a = 0; a <= (int64_t)(sqrt((double)n)); ++a) {
        int64_t remainder = n - a * a;
        int64_t b = (int64_t)(sqrt((double)remainder));
        if (b * b == remainder) {
            rep.a = a;
            rep.b = b;
            return rep;
        }
    }

    // Check for three squares
    for (a = 0; a <= (int64_t)(sqrt((double)n)); ++a) {
        int64_t remainder1 = n - a * a;
        for (int64_t b = 0; b <= (int64_t)(sqrt((double)remainder1)); ++b) {
            int64_t remainder2 = remainder1 - b * b;
            int64_t c = (int64_t)(sqrt((double)remainder2));
            if (c * c == remainder2) {
                rep.a = a;
                rep.b = b;
                rep.c = c;
                return rep;
            }
        }
    }

    // Use four squares
    for (a = 0; a <= (int64_t)(sqrt((double)n)); ++a) {
        int64_t remainder1 = n - a * a;
        for (int64_t b = 0; b <= (int64_t)(sqrt((double)remainder1)); ++b) {
            int64_t remainder2 = remainder1 - b * b;
            for (int64_t c = 0; c <= (int64_t)(sqrt((double)remainder2)); ++c) {
                int64_t d = (int64_t)(sqrt((double)(n - a * a - b * b - c * c)));
                if (d * d == (n - a * a - b * b - c * c)) {
                    rep.a = a;
                    rep.b = b;
                    rep.c = c;
                    rep.d = d;
                    return rep;
                }
            }
        }
    }

    // According to Lagrange's theorem, every number should have a representation
    return rep;
}

// CUDA Kernel to find representations
__global__ void findRepresentationsKernel(int64_t* numbers, Representation* representations, int total_numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_numbers)
        return;

    int64_t n = numbers[idx];
    representations[idx] = findRepresentationDevice(n);
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

// Function to process odd or even numbers on a specific GPU
void processGPU(int device, const std::vector<int64_t>& h_numbers_subset, std::vector<Representation>& h_representations_subset) {
    checkCudaError(cudaSetDevice(device), "Setting device");

    int64_t* d_numbers;
    Representation* d_representations;

    size_t numbers_size = h_numbers_subset.size() * sizeof(int64_t);
    size_t representations_size = h_numbers_subset.size() * sizeof(Representation);

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_numbers, numbers_size), "Allocating d_numbers");
    checkCudaError(cudaMalloc((void**)&d_representations, representations_size), "Allocating d_representations");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_numbers, h_numbers_subset.data(), numbers_size, cudaMemcpyHostToDevice), "Copying h_numbers_subset to d_numbers");

    // Determine block and grid sizes
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, findRepresentationsKernel, 0, 0);
    gridSize = (h_numbers_subset.size() + blockSize - 1) / blockSize;

    std::cout << "Processing GPU " << device << " (" << (device == 0 ? "odd" : "even") << " numbers):\n";
    std::cout << "  Total numbers: " << h_numbers_subset.size() << "\n";
    std::cout << "  Optimal Block Size: " << blockSize << "\n";
    std::cout << "  Grid Size: " << gridSize << "\n";

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");

    // Record the start event
    checkCudaError(cudaEventRecord(start, 0), "Recording start event");

    // Launch the kernel
    findRepresentationsKernel<<<gridSize, blockSize>>>(d_numbers, d_representations, h_numbers_subset.size());

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Launching findRepresentationsKernel");

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, 0), "Recording stop event");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");

    std::cout << "  Kernel execution completed in " << milliseconds / 1000.0f << " seconds.\n";

    // Copy results back to host
    checkCudaError(cudaMemcpy(h_representations_subset.data(), d_representations, representations_size, cudaMemcpyDeviceToHost), "Copying d_representations to h_representations_subset");

    // Clean up CUDA events and device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_numbers);
    cudaFree(d_representations);
}

int main() {
    const int64_t LOWER_BOUND = 1;
    const int64_t UPPER_BOUND = 1000000; // 1,000,000

    std::cout << "Lagrange's Four Square Theorem Representation Finder (CUDA Multi-GPU Simultaneous Execution with Parity)\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------\n";

    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Getting device count");

    if (deviceCount < 2) {
        std::cerr << "Error: At least two CUDA-capable GPUs are required.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Number of CUDA devices detected: " << deviceCount << "\n";

    // Host-side data
    std::vector<int64_t> h_numbers_odd;
    std::vector<int64_t> h_numbers_even;
    for (int64_t i = LOWER_BOUND; i <= UPPER_BOUND; ++i) {
        if (i % 2 == 1) {
            h_numbers_odd.push_back(i);
        } else {
            h_numbers_even.push_back(i);
        }
    }

    // Allocate host memory for representations
    std::vector<Representation> h_representations_odd(h_numbers_odd.size());
    std::vector<Representation> h_representations_even(h_numbers_even.size());

    // Start timing
    cudaEvent_t global_start, global_stop;
    checkCudaError(cudaEventCreate(&global_start), "Creating global start event");
    checkCudaError(cudaEventCreate(&global_stop), "Creating global stop event");

    checkCudaError(cudaEventRecord(global_start, 0), "Recording global start event");

    // Launch processing on each GPU in separate threads for simultaneous execution
    std::thread gpu_thread_odd(processGPU, 0, std::cref(h_numbers_odd), std::ref(h_representations_odd));
    std::thread gpu_thread_even(processGPU, 1, std::cref(h_numbers_even), std::ref(h_representations_even));

    // Wait for both GPU threads to finish
    gpu_thread_odd.join();
    gpu_thread_even.join();

    // Record the stop event
    checkCudaError(cudaEventRecord(global_stop, 0), "Recording global stop event");
    checkCudaError(cudaEventSynchronize(global_stop), "Synchronizing global stop event");

    // Calculate elapsed time
    float total_milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&total_milliseconds, global_start, global_stop), "Calculating total elapsed time");

    std::cout << "Total kernel execution across all GPUs completed in " << total_milliseconds / 1000.0f << " seconds.\n";

    // Destroy global CUDA events
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);

    // Combine results and save to file
    std::cout << "Saving representations to 'representations_cuda_parity_simultaneous.txt'...\n";
    std::ofstream outfile("representations_cuda_parity_simultaneous.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return EXIT_FAILURE;
    }

    // Write odd numbers
    for (size_t i = 0; i < h_numbers_odd.size(); ++i) {
        int64_t n = h_numbers_odd[i];
        Representation rep = h_representations_odd[i];
        outfile << n << " = " << rep.a << "^2";
        if (rep.b > 0 || rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.b << "^2";
        }
        if (rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.c << "^2";
        }
        if (rep.d > 0) {
            outfile << " + " << rep.d << "^2";
        }
        outfile << "\n";
    }

    // Write even numbers
    for (size_t i = 0; i < h_numbers_even.size(); ++i) {
        int64_t n = h_numbers_even[i];
        Representation rep = h_representations_even[i];
        outfile << n << " = " << rep.a << "^2";
        if (rep.b > 0 || rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.b << "^2";
        }
        if (rep.c > 0 || rep.d > 0) {
            outfile << " + " << rep.c << "^2";
        }
        if (rep.d > 0) {
            outfile << " + " << rep.d << "^2";
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Representations saved successfully to 'representations_cuda_parity_simultaneous.txt'.\n";

    // Optional: Display summary statistics
    int count1 = 0, count2 = 0, count3 = 0, count4 = 0;

    auto count_squares = [&](const std::vector<Representation>& reps) {
        for (const auto& rep : reps) {
            int squares_used = 0;
            if (rep.a > 0) squares_used++;
            if (rep.b > 0) squares_used++;
            if (rep.c > 0) squares_used++;
            if (rep.d > 0) squares_used++;
            switch (squares_used) {
                case 1: count1++; break;
                case 2: count2++; break;
                case 3: count3++; break;
                case 4: count4++; break;
                default: break;
            }
        }
    };

    count_squares(h_representations_odd);
    count_squares(h_representations_even);

    std::cout << "\nSummary Statistics:\n";
    std::cout << "-------------------\n";
    std::cout << "Numbers expressed as the sum of 1 square: " << count1 << "\n";
    std::cout << "Numbers expressed as the sum of 2 squares: " << count2 << "\n";
    std::cout << "Numbers expressed as the sum of 3 squares: " << count3 << "\n";
    std::cout << "Numbers expressed as the sum of 4 squares: " << count4 << "\n";

    return 0;
}
