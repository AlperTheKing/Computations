import mpmath
import time

if __name__ == '__main__':
    mpmath.mp.dps = 10000000  # Set decimal places for e

    start_time = time.time()
    
    e_digits = str(mpmath.nstr(mpmath.e, 10000000))

    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time} seconds")
    print(e_digits[:100])  # Print the first 100 digits of e
