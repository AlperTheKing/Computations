#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>

const int NUM_SHOTS = 50;
const int TARGET_POINTS = 20;
const long double TARGET_PROB = 0.02;
const long double Q_LOWER_BOUND = 50.0;
const long double Q_UPPER_BOUND = 1000.0;
const long double TOLERANCE = 1e-12;
const int MAX_ITERATIONS = 1000;

std::mutex mtx;

long double compute_probability(int target, const std::vector<long double>& p) {
    std::vector<long double> dp(target + 1, 0.0L);
    dp[0] = 1.0L;
    for (int shot = 0; shot < NUM_SHOTS; ++shot) {
        long double prob = p[shot];
        for (int i = std::min(shot + 1, target); i >= 1; --i) {
            dp[i] = dp[i] * (1.0L - prob) + dp[i - 1] * prob;
        }
        dp[0] *= (1.0L - prob);
    }
    return dp[target];
}

long double P_X_equals_target(long double q) {
    std::vector<long double> p(NUM_SHOTS, 0.0L);
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    int shots_per_thread = NUM_SHOTS / num_threads;
    int remaining_shots = NUM_SHOTS % num_threads;
    int current_shot = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = current_shot;
        int end = start + shots_per_thread + (t < remaining_shots ? 1 : 0);
        current_shot = end;
        threads.emplace_back([&p, q, start, end]() {
            for (int i = start; i < end; ++i) {
                p[i] = 1.0L - static_cast<long double>(i + 1) / q;
                if (p[i] < 0.0L) p[i] = 0.0L;
                if (p[i] > 1.0L) p[i] = 1.0L;
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    return compute_probability(TARGET_POINTS, p);
}

long double find_q_bisection() {
    long double q_low = Q_LOWER_BOUND;
    long double q_high = Q_UPPER_BOUND;
    long double q_mid = 0.0L;
    long double prob_mid = 0.0L;
    long double f_low = P_X_equals_target(q_low) - TARGET_PROB;
    long double f_high = P_X_equals_target(q_high) - TARGET_PROB;
    while (f_high > 0.0L && q_high < 1e12L) {
        q_high *= 2.0L;
        f_high = P_X_equals_target(q_high) - TARGET_PROB;
    }
    if (f_low < 0.0L || f_high > 0.0L) {
        std::cerr << "Bisection method failed to find suitable initial bounds." << std::endl;
        return -1.0L;
    }
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        q_mid = 0.5L * (q_low + q_high);
        prob_mid = P_X_equals_target(q_mid) - TARGET_PROB;
        if (std::abs(prob_mid) < TOLERANCE) {
            return q_mid;
        }
        if (prob_mid > 0.0L) {
            q_low = q_mid;
        } else {
            q_high = q_mid;
        }
        if ((q_high - q_low) < TOLERANCE) {
            return q_mid;
        }
    }
    std::cerr << "Bisection method did not converge within the maximum number of iterations." << std::endl;
    return q_mid;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    long double q = find_q_bisection();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long double> duration = end_time - start_time;
    if (q > 0.0L) {
        std::vector<long double> p(NUM_SHOTS, 0.0L);
        for (int i = 0; i < NUM_SHOTS; ++i) {
            p[i] = 1.0L - static_cast<long double>(i + 1) / q;
            if (p[i] < 0.0L) p[i] = 0.0L;
            if (p[i] > 1.0L) p[i] = 1.0L;
        }
        long double final_prob = compute_probability(TARGET_POINTS, p);
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Computed q: " << q << std::endl;
        std::cout << "P(X = " << TARGET_POINTS << ") = " << final_prob << std::endl;
        std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;
    } else {
        std::cerr << "Failed to compute q." << std::endl;
    }
    return 0;
}