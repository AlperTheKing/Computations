#include <stdio.h>
#include <stdint.h>

// Function to calculate the number of times a number is divisible by a prime p
int count_factors(int num, int p) {
    int count = 0;
    while (num % p == 0) {
        num /= p;
        count++;
    }
    return count;
}

// Function to compute the product modulo 10^100 (ignoring factors of 2 and 5)
void mod_product(int start, int end, int step, uint64_t result[2]) {
    const uint64_t MOD_LOW = 1000000000000000000ULL;  // 10^18 for low part
    const uint64_t MOD_HIGH = 1000000000000000000ULL; // 10^18 for high part

    uint64_t low = 1, high = 0;
    int total_twos = 0, total_fives = 0;

    for (int i = start; i <= end; i += step) {
        int num = i;

        // Count and remove factors of 2 and 5
        total_twos += count_factors(num, 2);
        total_fives += count_factors(num, 5);

        while (num % 2 == 0) num /= 2;
        while (num % 5 == 0) num /= 5;

        // Multiply and reduce modulo
        low = low * num;
        high = high * num + low / MOD_LOW;
        low = low % MOD_LOW;
        high = high % MOD_HIGH;
    }

    // Adjust the result by adding back the trailing zeros
    int trailing_zeros = total_twos < total_fives ? total_twos : total_fives;
    for (int i = 0; i < trailing_zeros; ++i) {
        low *= 10;
        high = high * 10 + low / MOD_LOW;
        low = low % MOD_LOW;
        high = high % MOD_HIGH;
    }

    result[0] = low;
    result[1] = high;
}

int main() {
    int start = 5;
    int end = 500;
    int step = 5;

    uint64_t result[2];
    mod_product(start, end, step, result);

    // Combine high and low parts into a string
    char result_str[37];
    snprintf(result_str, sizeof(result_str), "%018llu%018llu", result[1], result[0]);

    // Ensure the string is long enough
    int len = 36;
    if (len < 98) {
        printf("The number is too short to have a 98th digit.\n");
        return 1;
    }

    // Output the 98th digit from the end
    char result_digit = result_str[len - 98];
    printf("The 98th digit from the end is: %c\n", result_digit);

    return 0;
}
