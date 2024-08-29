#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000000000000000ULL  // 10^18

// Function to calculate the number of times a number is divisible by a prime p
int count_factors(int num, int p) {
    int count = 0;
    while (num % p == 0) {
        num /= p;
        count++;
    }
    return count;
}

// Function to compute the product modulo 10^18 (ignoring factors of 2 and 5)
unsigned long long mod_product(int start, int end, int step) {
    unsigned long long result = 1;
    int total_twos = 0, total_fives = 0;

    for (int i = start; i <= end; i += step) {
        int num = i;

        // Count and remove factors of 2 and 5
        total_twos += count_factors(num, 2);
        total_fives += count_factors(num, 5);

        while (num % 2 == 0) num /= 2;
        while (num % 5 == 0) num /= 5;

        result *= num;
        result %= MOD;  // Keep the result manageable
    }

    // Adjust the result by adding back the trailing zeros
    int trailing_zeros = total_twos < total_fives ? total_twos : total_fives;
    for (int i = 0; i < trailing_zeros; ++i) {
        result *= 10;
        result %= MOD;
    }

    return result;
}

int main() {
    int start = 5;
    int end = 500;
    int step = 5;

    // Compute the product modulo 10^18
    unsigned long long last_digits = mod_product(start, end, step);

    // Convert the result to a string to access the digits
    char result_str[19];
    snprintf(result_str, sizeof(result_str), "%018llu", last_digits);

    // The 98th digit from the end
    char result_digit = '0';
    int digit_length = strlen(result_str);

    if (digit_length >= 98) {
        result_digit = result_str[digit_length - 98];
    } else {
        int offset = 98 - digit_length;
        if (offset < 18) {
            result_digit = result_str[18 - offset];
        }
    }

    printf("The 98th digit from the end is: %c\n", result_digit);

    return 0;
}
