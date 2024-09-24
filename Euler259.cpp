#include <iostream>
#include <stack>
#include <string>
#include <set>
#include <vector>
#include <chrono>
#include <boost/rational.hpp>  // Boost Rational for accurate fractions
#include <boost/multiprecision/cpp_int.hpp>  // To handle large integers
#include <map>
#include <thread>
#include <mutex>
#include <fstream>  // For file output
#include <algorithm> // For sorting

using boost::rational;
using boost::multiprecision::cpp_int;
std::mutex mtx;  // Mutex for safely accessing shared data

// List of operators to use, including concatenation (denoted by empty string)
std::vector<std::string> operators = {"+", "-", "*", "/", ""};

// Helper function to apply an operator to two rational operands
rational<int> applyOperator(rational<int> a, rational<int> b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': 
            if (b != 0) {  // Avoid division by zero
                return a / b;
            } else {
                throw std::overflow_error("Division by zero");
            }
    }
    return 0;
}

// Function to evaluate an expression using std::stack for operands and operators
rational<int> evaluateExpression(const std::string& expr) {
    std::stack<rational<int>> values;  // Stack to store values as rational numbers
    std::stack<char> ops;        // Stack to store operators
    int num = 0;
    bool hasNum = false;

    for (size_t i = 0; i < expr.length(); ++i) {
        char ch = expr[i];

        // If digit, accumulate the number
        if (isdigit(ch)) {
            num = num * 10 + (ch - '0');
            hasNum = true;
        }

        // If we encounter an operator or the end of the expression
        if (!isdigit(ch) || i == expr.length() - 1) {
            if (hasNum) {
                values.push(rational<int>(num));  // Push accumulated number as a rational to the values stack
                num = 0;
                hasNum = false;
            }

            if (ch == '(') {
                ops.push(ch);  // Push opening parenthesis to operators stack
            } else if (ch == ')') {
                // Pop until we encounter an opening parenthesis
                while (!ops.empty() && ops.top() != '(') {
                    rational<int> b = values.top(); values.pop();
                    rational<int> a = values.top(); values.pop();
                    char op = ops.top(); ops.pop();
                    values.push(applyOperator(a, b, op));
                }
                ops.pop();  // Pop the '('
            } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
                // Apply the operator precedence (evaluate previous operators with higher precedence)
                while (!ops.empty() && ops.top() != '(' &&
                       ((ch == '+' || ch == '-') || (ch == '*' || ch == '/' && (ops.top() == '*' || ops.top() == '/')))) {
                    rational<int> b = values.top(); values.pop();
                    rational<int> a = values.top(); values.pop();
                    char op = ops.top(); ops.pop();
                    values.push(applyOperator(a, b, op));
                }
                ops.push(ch);  // Push current operator to stack
            }
        }
    }

    // Evaluate the remaining operations in the stack
    while (!ops.empty()) {
        rational<int> b = values.top(); values.pop();
        rational<int> a = values.top(); values.pop();
        char op = ops.top(); ops.pop();
        values.push(applyOperator(a, b, op));
    }

    return values.top();  // The final value on the stack is the result
}

// Recursive function to generate all possible operator combinations without splitting concatenated numbers
void generateOperatorPermutations(const std::string& digits, std::vector<std::string>& expressions, std::string currentExpr, int index) {
    if (index == digits.length() - 1) {
        expressions.push_back(currentExpr + digits[index]);  // Final digit
        return;
    }

    // Add each operator between the digits
    for (const auto& op : operators) {
        // If concatenating, continue without adding an operator
        if (op == "") {
            generateOperatorPermutations(digits, expressions, currentExpr + digits[index], index + 1);
        } else {
            generateOperatorPermutations(digits, expressions, currentExpr + digits[index] + op, index + 1);
        }
    }
}

// Recursive function to generate all possible valid ways to parenthesize an expression
std::set<std::string> generateAllParentheses(const std::string& expr) {
    std::set<std::string> result;

    // Base case: if the expression is just a number, return it
    if (expr.find_first_of("+-*/") == std::string::npos) {
        result.insert(expr);
        return result;
    }

    // Recursively split the expression at every operator and apply parentheses
    for (size_t i = 0; i < expr.length(); ++i) {
        if (!isdigit(expr[i])) {  // Find an operator
            // Divide the expression into left and right parts
            std::string leftExpr = expr.substr(0, i);
            std::string rightExpr = expr.substr(i + 1);
            std::string op(1, expr[i]);

            // Get all valid parentheses combinations for both left and right sub-expressions
            std::set<std::string> leftCombinations = generateAllParentheses(leftExpr);
            std::set<std::string> rightCombinations = generateAllParentheses(rightExpr);

            // Combine the left and right sub-expressions with the operator and add parentheses
            for (const std::string& left : leftCombinations) {
                for (const std::string& right : rightCombinations) {
                    result.insert("(" + left + op + right + ")");
                }
            }
        }
    }
    return result;
}

// Function to evaluate expressions in parallel and remove invalid (division by zero) expressions
void evaluateInThread(const std::vector<std::string>& expressions, std::vector<std::pair<cpp_int, std::string>>& results, std::set<cpp_int>& uniqueResults, int start, int end) {
    for (int i = start; i < end; ++i) {
        const std::string& expr = expressions[i];
        std::set<std::string> parenthesesResults = generateAllParentheses(expr);

        // Evaluate each expression
        for (const auto& parenthesizedExpr : parenthesesResults) {
            try {
                rational<int> result = evaluateExpression(parenthesizedExpr);

                // Ensure the result is an integer
                if (result.denominator() == 1 && result.numerator() > 0) {  // Only store positive integer results
                    std::lock_guard<std::mutex> lock(mtx);  // Ensure safe access to shared results
                    cpp_int integerResult = result.numerator();  // Convert to integer
                    if (uniqueResults.insert(integerResult).second) {  // Insert result and check if it's new
                        results.emplace_back(integerResult, parenthesizedExpr);
                    }
                }
            } catch (const std::overflow_error&) {
                // Skip invalid expressions that cause division by zero
                continue;
            }
        }
    }
}

// Main function to generate and evaluate all possible expressions with multithreading
int main() {
    const std::string digits = "123456789";

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate all operator permutations
    std::vector<std::string> expressions;
    generateOperatorPermutations(digits, expressions, "", 0);

    std::vector<std::pair<cpp_int, std::string>> results;  // To store results (integer, expression)
    std::set<cpp_int> uniqueResults;  // To ensure only unique results are added

    // Number of threads to use
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int numExpressionsPerThread = expressions.size() / numThreads;

    // Split the work between threads
    for (int t = 0; t < numThreads; ++t) {
        int start = t * numExpressionsPerThread;
        int end = (t == numThreads - 1) ? expressions.size() : start + numExpressionsPerThread;
        threads.emplace_back(evaluateInThread, std::ref(expressions), std::ref(results), std::ref(uniqueResults), start, end);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Sort results by the integer value
    std::sort(results.begin(), results.end());

    // Write results to result.txt
    std::ofstream outfile("result.txt");
    for (const auto& [num, expr] : results) {
        outfile << "Integer: " << num << "\tExpression: " << expr << "\n";
    }
    outfile.close();

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Output summary
    std::cout << "Total number of unique reachable numbers: " << results.size() << std::endl;
    cpp_int sumReachable = 0;
    for (const auto& [num, _] : results) {
        sumReachable += num;
    }
    std::cout << "Sum of all unique reachable integers: " << sumReachable << std::endl;
    std::cout << "Elapsed time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}