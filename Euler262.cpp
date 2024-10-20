// Complete C++ Code
#include <iostream>
#include <cmath>
#include <functional>
#include <boost/math/tools/minima.hpp>      // For finding local maxima/minima
#include <boost/math/tools/roots.hpp>       // For root finding
#include <boost/math/special_functions/lambert_w.hpp> // For Lambert W function
#include <boost/math/quadrature/gauss_kronrod.hpp> // For numerical integration

int main() {
    // Block 1
    auto h_function = [](double x, double y) {
        double h = 1000.0 * (5.0 - 5.0 / 1e6 * (x * x + y * y + x * y) + 125.0 / 1e4 * (x + y))
                   * std::exp(-std::abs(1.0 / 1e6 * (x * x + y * y) - 15.0 / 1e4 * (x + y) + 0.7));
        return h;
    };

    // Variables
    double u_l = 200.0 * std::sqrt(2.0);
    double u_r = 1400.0 * std::sqrt(2.0);

    // Finding the local maximum of h(y=0) between x = 0 and x = 1600
    auto h_y0 = [&](double x) { return -h_function(x, 0.0); }; // Negative for maximization
    boost::uintmax_t max_iter = 50;
    std::pair<double, double> result = boost::math::tools::brent_find_minima(h_y0, 0.0, 1600.0, std::numeric_limits<double>::digits, max_iter);
    double x_left = result.first;
    double f_min = -result.second;

    std::cout << "Local maximum of h(y=0) is at x = " << x_left << " with value = " << f_min << std::endl;

    // Block 3
    auto A = [](double u) {
        return 2.0 - 3.0 * u * u / 1e6 + std::sqrt(2.0) * u / 200.0;
    };

    auto B = [](double u) {
        return -u * u / 1e6 + 3.0 * std::sqrt(2.0) * u / 2000.0 - 0.7;
    };

    // Define h3(u, v)
    auto h3 = [&](double u, double v) {
        double h3_val = (A(u) - v * v / 1e6) * std::exp(B(u) - v * v / 1e6) * 2500.0;
        return h3_val;
    };

    // Find the maximum height
    auto h3_v0 = [&](double u) { return h3(u, 0.0); };

    boost::uintmax_t max_iter_h3 = 50;
    std::pair<double, double> result_h3 = boost::math::tools::brent_find_minima(
        [&](double u) { return -h3_v0(u); }, 0.0, 2000.0, std::numeric_limits<double>::digits, max_iter_h3);
    double height_u = result_h3.first;
    double height = h3_v0(height_u);

    std::cout << "Maximum height is at u = " << height_u << " with value = " << height << std::endl;

    // Block 2
    // Define f_v(u)
    auto f_v = [&](double u) {
        double exp_part = std::exp(A(u) - B(u));
        double lambert_input = height / 2500.0 * exp_part;
        double W = boost::math::lambert_w0(lambert_input);
        double fv = std::sqrt(A(u) - W) * 1000.0;
        return fv;
    };

    // Derivative of f_v(u) (approximated numerically)
    auto f_v_derivative = [&](double u) {
        double h = 1e-6;
        return (f_v(u + h) - f_v(u - h)) / (2 * h);
    };

    // Define the functions to find roots x_L and x_R
    auto func_L = [&](double u) {
        return u - f_v(u) / f_v_derivative(u) - u_l;
    };

    auto func_R = [&](double u) {
        return u - f_v(u) / f_v_derivative(u) - u_r;
    };

    // Find x_L and x_R using root-finding
    boost::uintmax_t max_iter_root = 50;
    double tol = 1e-6;

    // Finding x_L
    auto sol_L = boost::math::tools::toms748_solve(func_L, 387.0, 900.0, func_L(387.0), func_L(900.0),
                                                   [tol](double min, double max) { return std::abs(max - min) <= tol; });
    double x_L = (sol_L.first + sol_L.second) / 2.0;

    // Finding x_R
    auto sol_R = boost::math::tools::toms748_solve(func_R, 1200.0, 1828.0, func_R(1200.0), func_R(1828.0),
                                                   [tol](double min, double max) { return std::abs(max - min) <= tol; });
    double x_R = (sol_R.first + sol_R.second) / 2.0;

    std::cout << "x_L = " << x_L << ", x_R = " << x_R << std::endl;

    // Compute ans
    double ans = std::hypot(u_l - x_L, f_v(x_L)) + std::hypot(u_r - x_R, f_v(x_R));
    ans += 1536.0161445693104;

    std::cout << "ans = " << ans << std::endl;

    // Numerical integration
    auto int_f = [&](double u) {
        double derivative = f_v_derivative(u);
        return std::sqrt(derivative * derivative + 1.0);
    };

    double integral_result = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(
        int_f, x_L, x_R, 5, 1e-9);

    std::cout << "Integral result = " << integral_result << std::endl;

    // Optional plotting data output
    /*
    std::ofstream data_file("h3_height_data.txt");
    for (double u = 0.0; u <= 2000.0; u += 10.0) {
        for (double v = -1000.0; v <= 1000.0; v += 10.0) {
            if (std::abs(h3(u, v) - height) < 1e-2) {
                data_file << u << " " << v << std::endl;
            }
        }
    }
    data_file.close();
    */

    return 0;
}