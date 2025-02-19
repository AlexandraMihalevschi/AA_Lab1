import time
import matplotlib.pyplot as plt
from functools import lru_cache
import sys
import numpy as np

sys.set_int_max_str_digits(10000)

# 1. Naive Recursive (Exponential Complexity)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# 2. Memoized Recursive (Top-Down Dynamic Programming)
@lru_cache(None)
def fib_memoized(n):
    if n <= 1:
        return n
    return fib_memoized(n-1) + fib_memoized(n-2)

# 3. Bottom-Up Dynamic Programming (Tabulation)
def fib_bottom_up(n):
    if n <= 1:
        return n
    dp = [0, 1] + [0] * (n-1)
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 4. Matrix Exponentiation
def matrix_mult(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]

def matrix_power(F, n):
    if n == 1:
        return F
    if n % 2 == 0:
        half = matrix_power(F, n // 2)
        return matrix_mult(half, half)
    else:
        return matrix_mult(F, matrix_power(F, n - 1))

def fib_matrix(n):
    if n == 0:
        return 0
    F = [[1, 1], [1, 0]]
    result = matrix_power(F, n - 1)
    return result[0][0]

# 5. Iterative Approach (Loop-Based)
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b

# 6. Binet Formula (Closed-Form Expression)
def fib_binet(n):
    if n > 70:
        raise ValueError("Binet formula is not accurate for n > 70 due to precision issues.")
    sqrt5 = np.sqrt(5)
    phi = (1 + sqrt5) / 2
    return round((phi**n - (-1/phi)**n) / sqrt5)


# Performance measurement
def measure_time(fib_function, ns):
    times = []
    for n in ns:
        start_time = time.time()
        fib_function(n)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        print(f"Time for {fib_function.__name__}({n}): {elapsed_time:.6f} seconds")  # Only print time
    return times

# Predefined Fibonacci numbers for testing
fib_numbers_recursive = [5, 8, 13, 21, 34]
fib_numbers_non_recursive = [5, 21, 89, 233, 987, 1597, 2584, 4181, 10946, 17711, 28657]

methods = {
    "Recursive": fib_recursive,
    "Memoized": fib_memoized,
    "Bottom-Up": fib_bottom_up,
    "Matrix Power": fib_matrix,
    "Iterative": fib_iterative,
    "Binet": fib_binet
}

results = {}

# Measure time for recursive methods (including Binet for smaller n)
for name, func in [("Recursive", fib_recursive),
                   ("Memoized", fib_memoized),
                   ("Binet", fib_binet)]:
    print(f"\nTesting {name} method:")
    results[name] = measure_time(func, fib_numbers_recursive)

# Measure time for non-recursive methods
for name, func in [("Bottom-Up", fib_bottom_up),
                   ("Matrix Power", fib_matrix),
                   ("Iterative", fib_iterative)]:
    print(f"\nTesting {name} method:")
    results[name] = measure_time(func, fib_numbers_non_recursive)

# Plot individual performance
for name, times in results.items():
    plt.figure()
    ns = fib_numbers_recursive if name in ["Recursive", "Memoized", "Binet"] else fib_numbers_non_recursive
    plt.plot(ns, times, marker='o', linestyle='-', label=name)  # Connect points with a line
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title(f"Performance of {name} Method")
    plt.legend()
    plt.grid()
    plt.show()

# Plot combined performance
plt.figure(figsize=(10, 6))
for name, times in results.items():
    ns = fib_numbers_recursive if name in ["Recursive", "Memoized", "Binet"] else fib_numbers_non_recursive
    plt.plot(ns, times, marker='o', linestyle='-', label=name)  # Connect points with a line
plt.xlabel("n")
plt.ylabel("Time (s)")
plt.title("Performance Comparison of Fibonacci Algorithms Including Binet Formula")
plt.legend()
plt.grid()
plt.yscale("log")
plt.show()
