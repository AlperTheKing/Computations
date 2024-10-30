#include <symengine/expression.h>
#include <symengine/functions.h>
#include <symengine/symbol.h>
#include <symengine/real_double.h>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using namespace SymEngine;

// Function to check if a number is a perfect square
bool isPerfectSquare(int num) {
    int sqrtVal = static_cast<int>(std::sqrt(num));
    return (sqrtVal * sqrtVal == num);
}

// Function to determine if an expression can be denested
bool canBeDenested(int x, int y, int z) {
    // Construct the expression sqrt(x + sqrt(y) + sqrt(z)) using SymEngine
    RCP<const Basic> expr_x = integer(x);
    RCP<const Basic> expr_y = integer(y);
    RCP<const Basic> expr_z = integer(z);

    // Create the inner square root terms: sqrt(y) and sqrt(z)
    RCP<const Basic> sqrt_y = sqrt(expr_y);
    RCP<const Basic> sqrt_z = sqrt(expr_z);

    // Create the complete expression: x + sqrt(y) + sqrt(z)
    RCP<const Basic> inner_expr = add(add(expr_x, sqrt_y), sqrt_z);
    RCP<const Basic> nested_expr = sqrt(inner_expr);

    // Expand the expression to see if it can be simplified
    RCP<const Basic> expanded_expr = expand(nested_expr);

    // Debugging information to see what is happening
    std::cout << "Original expr: " << *nested_expr << "\n";
    std::cout << "Expanded expr: " << *expanded_expr << "\n";

    // Check if the expression was expanded to something different
    return !eq(*nested_expr, *expanded_expr);
}

// Function to compute F(n)
int F(int n) {
    std::set<std::vector<int>> uniqueTerms;

    for (int x = 1; x <= n; ++x) {
        for (int y = 1; y <= n; ++y) {
            if (isPerfectSquare(y)) continue; // Skip perfect squares for y
            for (int z = 1; z <= n; ++z) {
                if (isPerfectSquare(z)) continue; // Skip perfect squares for z

                if (canBeDenested(x, y, z)) {
                    // Store x, y, z as a sorted vector to avoid duplicates with different permutations
                    std::vector<int> term = {x, y, z};
                    std::sort(term.begin(), term.end());
                    uniqueTerms.insert(term);
                }
            }
        }
    }

    return uniqueTerms.size();
}

int main() {
    int n = 10; // Change to other values like 15, 20, etc., for testing
    int result = F(n);
    std::cout << "F(" << n << ") = " << result << std::endl;
    return 0;
}