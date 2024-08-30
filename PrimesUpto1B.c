#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>

bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int compare_ints(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int main() {
    int *primes = NULL;
    int count = 0;
    int capacity = 10;
    int allocation_failure = 0; // Variable to track allocation failure

    primes = (int*)malloc(capacity * sizeof(int));

    if (primes == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    clock_t start_time = clock();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < 1000000000; ++i) {
        if (is_prime(i)) {
            #pragma omp critical
            {
                if (count == capacity) {
                    int *temp = (int*)realloc(primes, 2 * capacity * sizeof(int));
                    if (temp == NULL) {
                        allocation_failure = 1; // Set failure flag
                    } else {
                        primes = temp;
                        capacity *= 2;
                    }
                }
                if (!allocation_failure) {
                    primes[count++] = i;
                }
            }
        }
    }

    if (allocation_failure) {
        fprintf(stderr, "Memory allocation failed during processing\n");
        free(primes);
        return 1;
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    qsort(primes, count, sizeof(int), compare_ints);

    printf("%d prime numbers found\n", count);
    printf("First 10 prime numbers: ");
    for (int i = 0; i < 10 && i < count; ++i) {
        printf("%d ", primes[i]);
    }
    printf("\n");

    printf("Last 10 prime numbers: ");
    for (int i = count - 10; i < count; ++i) {
        printf("%d ", primes[i]);
    }
    printf("\n");

    printf("%.2f seconds elapsed.\n", elapsed_time);

    free(primes);
    return 0;
}