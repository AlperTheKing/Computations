#include <iostream>
#include <cmath>
#include <limits>

int main() {
    double max_value = -std::numeric_limits<double>::infinity();
    int max_n = 0;

    double current_value = 0;
    double factorial = 1;

    for (int n = 1; n <= 100; ++n) { // Limit n to 100 to prevent overflow
        factorial *= n;  // Calculate n!
        current_value = (std::pow(7, n) + std::pow(50, n)) / factorial;

        if (current_value > max_value) {
            max_value = current_value;
            max_n = n;
        } else {
            // Since the values are starting to decrease, we break out of the loop
            break;
        }
    }

    std::cout << "The value of n that maximizes the expression is: " << max_n << std::endl;
    std::cout << "The maximum value of the expression is: " << max_value << std::endl;

    return 0;
}