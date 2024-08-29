#include <iostream>

// Function to count the number of times a prime p divides n!
int count_p_in_factorial(int n, int p) {
    int count = 0;
    while (n > 0) {
        n /= p;
        count += n;
    }
    return count;
}

// Function to count the number of times a prime p divides n * (n+1) * ... * m
int count_p_in_product(int n, int m, int p) {
    return count_p_in_factorial(m, p) - count_p_in_factorial(n-1, p);
}

int main() {
    int start = 913;
    int end = 1385;
    int prime1 = 3;
    int prime2 = 7;

    // Count the highest powers of 3 and 7 in the product
    int count3 = count_p_in_product(start, end, prime1);
    int count7 = count_p_in_product(start, end, prime2);

    // The highest power of 21 is the minimum of count3 and count7
    int k = std::min(count3, count7);

    std::cout << "The largest k such that the expression is an integer is: " << k << std::endl;

    return 0;
}