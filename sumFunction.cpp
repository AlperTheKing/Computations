// sumFunction_simplified.cpp

#include <iostream>
#include <iomanip>
#include <chrono>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;

int main() {
    // Use cpp_int for large integer N
    cpp_int N("1000000000000000");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute S = 1 - 1 / (2N + 1)
    cpp_int denominator = 2 * N + 1;

    // Use high-precision float for the division
    cpp_dec_float_50 fraction = cpp_dec_float_50(1) / denominator.convert_to<cpp_dec_float_50>();

    cpp_dec_float_50 sum = 1 - fraction;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Set precision for output
    std::cout << std::setprecision(15);
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}