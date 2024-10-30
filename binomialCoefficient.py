import sys
import threading

def count_p_in_binomial(N, P):
    # Convert N to base P digits (least significant digit first)
    N_digits = []
    temp_N = N
    while temp_N > 0:
        N_digits.append(temp_N % P)
        temp_N //= P

    # Initialize DP table
    from collections import defaultdict
    dp = {0: 1}  # Key: exponent L, Value: count

    # Process digits from most significant to least significant
    for pos in reversed(range(len(N_digits))):
        N_digit = N_digits[pos]
        new_dp = defaultdict(int)
        for L, count in dp.items():
            # For possible K_digit values from 0 to N_digit
            for K_digit in range(N_digit + 1):
                N_minus_K_digit = N_digit - K_digit
                carry = 0
                if K_digit + N_minus_K_digit >= P:
                    carry = 1
                new_L = L + carry
                new_dp[new_L] += count
        dp = new_dp

    # Build the result array
    max_L = max(dp.keys())
    result = [dp.get(L, 0) for L in range(max_L + 1)]
    return result

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    Ns = []
    Ps = []
    for _ in range(T):
        N_str, P_str = sys.stdin.readline().split()
        N, P = int(N_str), int(P_str)
        Ns.append(N)
        Ps.append(P)
    results = []
    for N, P in zip(Ns, Ps):
        result = count_p_in_binomial(N, P)
        results.append(' '.join(map(str, result)))
    print('\n'.join(results))

if __name__ == "__main__":
    threading.Thread(target=main).start()