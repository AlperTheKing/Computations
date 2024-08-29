#include <iostream>
#include <gmp.h>

int count_trailing_zeros(mpz_t n) {
    int count = 0;
    mpz_t rem;
    mpz_init(rem);

    while (true) {
        mpz_mod_ui(rem, n, 10); // rem = n % 10
        if (mpz_cmp_ui(rem, 0) != 0) {
            break;
        }
        count++;
        mpz_div_ui(n, n, 10); // n = n / 10
    }

    mpz_clear(rem);
    return count;
}

int main() {
    mpz_t result;
    mpz_init(result);

    // Calculate 101^100
    mpz_ui_pow_ui(result, 101, 100);

    // Subtract 1 from the result
    mpz_sub_ui(result, result, 1);

    // Print the result (optional, may be very large)
    // gmp_printf("%Zd\n", result);

    // Count trailing zeros
    int trailing_zeros = count_trailing_zeros(result);

    std::cout << "Number of trailing zeros: " << trailing_zeros << std::endl;

    mpz_clear(result);
    return 0;
}
