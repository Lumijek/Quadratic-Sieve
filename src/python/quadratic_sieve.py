import numpy as np
import random
import logging
import time
from math import ceil, floor, exp, log, isqrt
from sympy import sqrt
from pprint import pprint
from line_profiler import LineProfiler


# -------------------------------------------------------------------
# Configure Logging
# -------------------------------------------------------------------
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Global Known Primes for Primality Testing
# -------------------------------------------------------------------
_known_primes = [2, 3]

def init_known_primes(limit=1000):
    """
    Initialize the known primes up to 'limit' using the 'is_prime' test.
    This helps optimize primality checks for smaller numbers.
    
    Args:
        limit (int): The upper bound to search for prime numbers.
    """
    global _known_primes
    # Generate primes from 5 to limit, skipping even numbers
    _known_primes += [x for x in range(5, limit, 2) if is_prime(x)]
    logger.info("Initialized _known_primes up to %d. Total known primes: %d", limit, len(_known_primes))

# -------------------------------------------------------------------
# Number Theory Utilities
# -------------------------------------------------------------------
def gcd(a, b):
    """
    Compute the GCD of two integers a and b using Euclid's Algorithm.
    
    Args:
        a (int): First integer.
        b (int): Second integer.
    
    Returns:
        int: Greatest Common Divisor of a and b.
    """
    a, b = abs(a), abs(b)
    while a:
        a, b = b % a, a
    return b

def legendre(n, p):
    """
    Compute the Legendre symbol (n/p).
    
    Args:
        n (int): The numerator.
        p (int): The prime denominator.
    
    Returns:
        int: 1 if n is a quadratic residue mod p,
             -1 if n is a non-quadratic residue mod p,
              0 if n ≡ 0 mod p.
    """
    val = pow(n, (p - 1) // 2, p)
    return val - p if val > 1 else val

def jacobi(a, m):
    """
    Compute the Jacobi symbol (a/m).
    
    Args:
        a (int): The numerator.
        m (int): The odd integer denominator.
    
    Returns:
        int: 1 if a is a QR modulo each prime factor of m,
             -1 if a is a non-QR for at least one prime factor,
              0 if a and m are not coprime.
    """
    a = a % m
    t = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if m % 8 in [3, 5]:
                t = -t
        a, m = m, a
        if a % 4 == 3 and m % 4 == 3:
            t = -t
        a %= m
    return t if m == 1 else 0

# -------------------------------------------------------------------
# Primality Testing (Miller-Rabin)
# -------------------------------------------------------------------
def _try_composite(a, d, n, s):
    """
    Internal helper for the Miller-Rabin primality test.
    
    Args:
        a (int): Base to test.
        d (int): The odd part of n-1.
        n (int): The number to test.
        s (int): The exponent of 2 in n-1.
    
    Returns:
        bool: True if n is definitely composite, False otherwise.
    """
    if pow(a, d, n) == 1:
        return False
    for i in range(s):
        if pow(a, 2**i * d, n) == n - 1:
            return False
    return True  # n is definitely composite

def is_prime(n, _precision_for_huge_n=16):
    """
    Miller-Rabin primality test with specific bases for certain ranges.
    
    Args:
        n (int): The number to test for primality.
        _precision_for_huge_n (int): Number of bases to test for very large n.
    
    Returns:
        bool: True if n is likely prime, False if composite.
    """
    if n in _known_primes:
        return True
    if any((n % p) == 0 for p in _known_primes) or n in (0, 1):
        return n in _known_primes  # ensures 2 or 3 recognized as prime, others not

    d, s = n - 1, 0
    while d % 2 == 0:
        d >>= 1
        s += 1

    # Check small ranges with small sets of bases
    if n < 1373653:
        return not any(_try_composite(a, d, n, s) for a in (2, 3))
    if n < 25326001:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5))
    if n < 118670087467:
        if n == 3215031751:
            return False
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7))
    if n < 2152302898747:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11))
    if n < 3474749660383:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13))
    if n < 341550071728321:
        return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17))

    # Otherwise, fall back to testing with known primes
    return not any(_try_composite(a, d, n, s) 
                   for a in _known_primes[:_precision_for_huge_n])

# -------------------------------------------------------------------
# Factorization (Brent's Method)
# -------------------------------------------------------------------
def brent(N):
    """
    Brent's factorization algorithm (variation of Pollard's Rho).
    Returns a non-trivial factor of N.
    
    Args:
        N (int): The composite number to factor.
    
    Returns:
        int: A non-trivial factor of N.
    """
    if N % 2 == 0:
        return 2

    y = random.randint(1, N - 1)
    c = random.randint(1, N - 1)
    m = random.randint(1, N - 1)

    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y % N + c) % N
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y % N + c) % N
                q = (q * abs(x - y)) % N
            g = gcd(q, N)
            k += m
        r <<= 1

    if g == N:
        while True:
            ys = (ys * ys % N + c) % N
            g = gcd(abs(x - ys), N)
            if g > 1:
                break
    return g

def factorise(n, factors):
    """
    Factorize 'n' and accumulate found factors in 'factors'.
    
    Args:
        n (int): The number to factorize.
        factors (list): The list to store found factors.
    
    Returns:
        int: The remaining part of n after factorization.
    """
    if(n < 0):
        factors.append(-1)
        n = -n
    rem = n
    while True:
        if is_prime(rem):
            factors.append(rem)
            break

        f = brent(rem)
        # If the factor is the same as remainder, try again
        while f == rem:
            f = brent(rem)

        if f and f < rem:
            if is_prime(f):
                factors.append(f)
                rem //= f
            else:
                rem_f = factorise(f, factors)
                rem = (rem // f) * rem_f
                # Remove rem_f if it was appended
                if rem_f in factors:
                    factors.remove(rem_f)
        else:
            break
    return rem

# -------------------------------------------------------------------
# Tonelli-Shanks (Modular Square Root)
# -------------------------------------------------------------------
def tonelli_shanks(a, p):
    """
    Solve x^2 ≡ a (mod p) for x.
    Returns (x, p - x).
    
    Args:
        a (int): The quadratic residue.
        p (int): The prime modulus.
    
    Returns:
        tuple: A pair of solutions (x, p - x).
    """
    a %= p
    if p % 8 in [3, 7]:
        x = pow(a, (p+1)//4, p)
        return x, p - x

    if p % 8 == 5:
        x = pow(a, (p+3)//8, p)
        if pow(x, 2, p) != a % p:
            x = (x * pow(2, (p-1)//4, p)) % p
        return x, p - x

    # General Tonelli-Shanks
    d = 2
    symb = 0
    while symb != -1:
        symb = jacobi(d, p)
        d += 1
    d -= 1

    n = p - 1
    s = 0
    while n % 2 == 0:
        n //= 2
        s += 1
    t = n
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        i1 = pow(2, s - 1 - i)
        i2 = (A * pow(D, m, p)) % p
        i3 = pow(i2, i1, p)
        if i3 == p - 1:
            m += pow(2, i)
    x = (pow(a, (t+1)//2, p) * pow(D, m//2, p)) % p
    return x, p - x

# -------------------------------------------------------------------
# Gaussian Elimination and Null Space Extraction
# -------------------------------------------------------------------
def gauss_elim(x):
    """
    Perform Gaussian elimination on the binary matrix 'x' over GF(2).
    It transforms the matrix into its row-echelon form and records the pivot columns.

    Args:
        x (np.ndarray): A binary matrix (containing 0 or 1).

    Returns:
        tuple:
            - np.ndarray: Row-echelon form of the matrix (in-place, dtype=int8).
            - list: Sorted list of pivot column indices.
    """
    # Step 1: Convert to bool for faster XOR
    x = x.astype(bool, copy=False)

    n, m = x.shape
    marks = []
    
    for i in range(n):
        row = x[i]
        ones = np.flatnonzero(row)
        if ones.size == 0:
            continue
        
        pivot = ones[0]
        marks.append(pivot)
        
        mask = x[:, pivot].copy()
        mask[i] = False
        
        x[mask] ^= row
    
    # Step 3: Convert back to int8 (0 or 1)
    return x.astype(np.int8, copy=False), sorted(marks)


def find_null_space_GF2(reduced_matrix, pivot_rows):
    """
    Find null space vectors of the reduced binary matrix over GF(2).
    
    Args:
        reduced_matrix (np.ndarray): The row-echelon form of the matrix after Gaussian elimination.
        pivot_rows (list): List of pivot row indices obtained from Gaussian elimination.
    
    Returns:
        np.ndarray: Array of null space vectors.
    """
    n, m = reduced_matrix.shape
    nulls = []
    free_rows = [row for row in range(n) if row not in pivot_rows]
    k = 0
    for row in free_rows:
        ones = np.where(reduced_matrix[row] == 1)[0]
        null = np.zeros(n)
        null[row] = 1

        # Vectorized version of the nested loop
        mask = np.isin(np.arange(n), pivot_rows)  # Create a mask for pivot rows
        relevant_cols = reduced_matrix[:, ones]  # Extract relevant columns (matching `ones`)
        matching_rows = np.any(relevant_cols == 1, axis=1)  # Rows with `1` in `ones` columns
        null[mask & matching_rows] = 1  # Update `null` only for matching pivot rows

        nulls.append(null)
        k += 1
        if k == 5:
            break

    nulls = np.asarray(nulls, dtype=np.int8)
    return nulls

# -------------------------------------------------------------------
# Prime Sieve
# -------------------------------------------------------------------
def prime_sieve(n):
    """
    Return a list of primes up to 'n' using the Sieve of Eratosthenes.
    
    Args:
        n (int): The upper limit to sieve for primes.
    
    Returns:
        list: List of prime numbers up to 'n'.
    """
    sieve_array = np.ones((n+1,), dtype=bool)
    sieve_array[0], sieve_array[1] = False, False
    for i in range(2, int(n**0.5) + 1):
        if sieve_array[i]:
            sieve_array[i*2 :: i] = False
    return np.where(sieve_array)[0].tolist()

# -------------------------------------------------------------------
# Quadratic Sieve Helper Functions
# -------------------------------------------------------------------
def poly(t, n):
    """
    Polynomial used in the Quadratic Sieve: f(t) = t^2 - n.
    
    Args:
        t (int): The variable.
        n (int): The number to factor.
    
    Returns:
        int: The value of the polynomial at t.
    """
    return t*t - n  # Equivalent to pow(t, 2) but slightly faster

def find_b(N):
    """
    Typical heuristic to determine the factor base bound B.
    
    Args:
        N (int): The number to factor.
    
    Returns:
        int: The bound B for the factor base.
    """
    x = ceil(exp(sqrt(0.5 * log(N) * log(log(N))))) + 1
    return x // 16

def get_smooth_b(N, B):
    """
    Build the factor base: primes p <= B where (N/p) = 1.
    
    Args:
        N (int): The number to factor.
        B (int): The bound for the factor base.
    
    Returns:
        list: The factor base as a list of primes.
    """
    primes = prime_sieve(B)  # Generate primes up to B
    factor_base = [2]  # Always include 2 in the factor base
    for p in primes[1:]:  # Skip 2 and check for quadratic residues
        if legendre(N, p) == 1:
            factor_base.append(p)
    return factor_base

# -------------------------------------------------------------------
# STEP 1: Decide Bound B
# -------------------------------------------------------------------
def decide_bound(N, B=None):
    """
    Decide on the bound B if none is provided using a heuristic.
    
    Args:
        N (int): The number to factor.
        B (int, optional): The bound for the factor base. If None, compute using heuristic.
    
    Returns:
        int: The bound B for the factor base.
    """
    if B is None:
        B = find_b(N)
    logger.info("Using B = %d", B)
    return B

# -------------------------------------------------------------------
# STEP 2: Build Factor Base
# -------------------------------------------------------------------
def build_factor_base(N, B):
    """
    Build the factor base for the Quadratic Sieve.
    
    Args:
        N (int): The number to factor.
        B (int): The bound for the factor base.
    
    Returns:
        list: The factor base as a list of primes.
    """
    fb = get_smooth_b(N, B)
    logger.info("Factor base size: %d", len(fb) + 1) # +1 for the -1
    return fb

# -------------------------------------------------------------------
# STEP 3: Sieve Phase - Find Potential Smooth Values
# -------------------------------------------------------------------
def sieve_interval(N, factor_base, I_multiplier=7500):
    """
    Sieve to find potential smooth values in the interval [base, base+I).
    
    Args:
        N (int): The number to factor.
        factor_base (list): The factor base primes.
        I_multiplier (int): Multiplier to determine the interval size based on factor base size.
    
    Returns:
        tuple: 
            - int: The base offset (floor(sqrt(N)) + 1).
            - int: The length of the interval.
            - list: The array of partially factored values.
    """
    I = len(factor_base) * I_multiplier  # Determine interval size
    base = floor(sqrt(N))  # Starting point for sieving

    half_I = I // 2
    # Evaluate the polynomial f(t) = t^2 - N over the interval
    x_values = [base + x for x in range(-half_I, half_I)]
    sieve_values = [poly(x, N) for x in x_values]

    # Remove powers of 2 from each sieve value
    for i in range(I):
        while sieve_values[i] % 2 == 0:
            sieve_values[i] //= 2
    # Remove powers of other primes in the factor base
    for p in factor_base[1:]:  # Skip 2 as it's already handled
        root1, root2 = tonelli_shanks(N, p)  # Find square roots of N mod p
        a = (root1 - base + half_I) % p  # Adjust roots to sieve offsets
        b = (root2 - base + half_I) % p
        for r in [a, b]:
            while r < I:
                while sieve_values[r] % p == 0:
                    sieve_values[r] //= p
                r += p

    return base, I, sieve_values, x_values

# -------------------------------------------------------------------
# STEP 4: Build Exponent Matrix from Smooth Values
# -------------------------------------------------------------------
def build_exponent_matrix(N, base, I, sieve_values, factor_base, x_values, T=1):
    """
    Build the exponent matrix for the Quadratic Sieve.
    
    Args:
        N (int): The number to factor.
        base (int): The base offset for sieving.
        I (int): The interval length.
        sieve_values (list): The array of partially factored sieve values.
        factor_base (list): The factor base primes.
        T (int): Threshold to determine when to stop collecting relations.
    
    Returns:
        tuple: 
            - list: The exponent matrix.
            - list: The list of relation identifiers.
            - list: The list of factorizations corresponding to each relation.
    """
    matrix = []
    relations = []
    factorizations = []
    fb_len = len(factor_base) + 1 # +1 for the -1
    zero_row = [0] * fb_len

    for i_offset in range(I):
        if sieve_values[i_offset] == 1 or sieve_values[i_offset] == -1:  # Check if the value is fully smooth over the factor base
            if len(relations) == fb_len + T:
                break

            row = zero_row.copy()
            value = poly(x_values[i_offset], N)  # Compute f(t) = t^2 - N

            # Fully factorize the value using the factor base
            local_factors = []
            factorise(value, local_factors)
            local_factors.sort()

            # Count each prime factor modulo 2 for the exponent matrix
            counts = {}
            for fac in local_factors:
                counts[fac] = counts.get(fac, 0) + 1

            for idx, prime in enumerate(factor_base):
                row[idx] = counts.get(prime, 0) % 2  # Record exponent modulo 2

            matrix.append(row)  # Add the row to the exponent matrix
            relations.append(x_values[i_offset])  # Record the relation identifier
            factorizations.append(local_factors)  # Store the factorization

    logger.info("Number of smooth relations: %d", len(relations))
    return matrix, relations, factorizations

# -------------------------------------------------------------------
# STEP 5: Solve for Dependencies in GF(2)
# -------------------------------------------------------------------
def solve_dependencies(matrix):
    """
    Perform linear algebra over GF(2) to find dependencies among relations.
    
    Args:
        matrix (list): The exponent matrix as a list of lists.
    
    Returns:
        np.ndarray: Array of dependency vectors.
    """
    logger.info("Solving linear system in GF(2).")
    matrix = np.array(matrix).T

    reduced_matrix, marks = gauss_elim(matrix)  # Perform Gaussian elimination
    null_basis = find_null_space_GF2(reduced_matrix.T, marks)  # Find null space vectors
    return null_basis

# -------------------------------------------------------------------
# STEP 6: Attempt to Extract Factors
# -------------------------------------------------------------------
def extract_factors(N, relations, factorizations, dep_vectors):
    """
    Use the dependency vectors to attempt factor extraction.
    
    Args:
        N (int): The number to factor.
        relations (list): List of relation identifiers.
        factorizations (list): List of factorizations corresponding to each relation.
        dep_vectors (np.ndarray): Array of dependency vectors.
    
    Returns:
        tuple: A pair of factors (p, q) if found, otherwise (0, 0).
    """
    for r in dep_vectors:
        prod_left = 1
        prod_right = 1
        for idx, bit in enumerate(r):
            if bit == 1:
                prod_left *= relations[idx]  # Multiply corresponding relation values
                for fac in factorizations[idx]:
                    prod_right *= fac  # Multiply corresponding factors

        sqrt_right = isqrt(prod_right)  # Compute integer square root
        prod_left = prod_left % N
        sqrt_right = sqrt_right % N
        factor_candidate = gcd(N, prod_left - sqrt_right)  # Compute GCD to find a non-trivial factor
        if factor_candidate not in (1, N):
            other_factor = N // factor_candidate
            logger.info("Found factors: %d, %d", factor_candidate, other_factor)
            return factor_candidate, other_factor

    return 0, 0  # No factors found

# -------------------------------------------------------------------
# Main Quadratic Sieve Function
# -------------------------------------------------------------------
def quadratic_sieve(N, B=None):
    """
    Perform the Quadratic Sieve on N.
    Splits the entire process into smaller steps.
    Logs timing for each stage.
    
    Args:
        N (int): The number to factor.
        B (int, optional): The bound for the factor base. If None, it is decided using a heuristic.
    
    Returns:
        tuple: A pair of factors (p, q) if found, otherwise (0, 0).
    """
    overall_start = time.time()
    logger.info("========== Quadratic Sieve V2 Start ==========")
    logger.info("Factoring N = %d", N)

    # Step 1: Decide Bound
    step_start = time.time()
    B = decide_bound(N, B)
    step_end = time.time()
    logger.info("Step 1 (Decide Bound) took %.3f seconds", step_end - step_start)

    # Step 2: Build Factor Base
    step_start = time.time()
    factor_base = build_factor_base(N, B)
    step_end = time.time()
    logger.info("Step 2 (Build Factor Base) took %.3f seconds", step_end - step_start)

    # Step 3: Sieve Phase
    step_start = time.time()
    base, I, sieve_vals, x_values = sieve_interval(N, factor_base)
    step_end = time.time()
    logger.info("Step 3 (Sieve Interval) took %.3f seconds", step_end - step_start)

    # Step 4: Build Exponent Matrix
    step_start = time.time()
    matrix, relations, factorizations = build_exponent_matrix(N, base, I, sieve_vals, factor_base, x_values, T=1)
    step_end = time.time()
    logger.info("Step 4 (Build Exponent Matrix) took %.3f seconds", step_end - step_start)

    if len(matrix) < len(factor_base) + 1:
        logger.warning("Not enough smooth relations found. Try increasing the sieve interval.")
        return 0, 0

    # Step 5: Solve for Dependencies (GF(2))
    step_start = time.time()
    dep_vectors = solve_dependencies(np.array(matrix))
    step_end = time.time()
    logger.info("Step 5 (Solve Dependencies) took %.3f seconds", step_end - step_start)


    # Step 6: Attempt to Extract Factors
    step_start = time.time()
    f1, f2 = extract_factors(N, relations, factorizations, dep_vectors)
    step_end = time.time()
    logger.info("Step 6 (Extract Factors) took %.3f seconds", step_end - step_start)
    if f1 and f2:
        logger.info("Quadratic Sieve successful: %d * %d = %d", f1, f2, N)
    else:
        logger.warning("No non-trivial factors found with the current settings.")

    overall_end = time.time()
    logger.info("Total time for Quadratic Sieve: %.3f seconds", overall_end - overall_start)
    logger.info("========== Quadratic Sieve End ==========")


    return f1, f2

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize small primes list
    init_known_primes(limit=10000)

    # Example composite numbers
    #N = 80672394923 * 16319916311
    #N = 87463  # Smaller example for testing
    N = 110945531268719200260254771214978881
    N = 867626227567916279 * 970373053360845209
    #N = 87463

    # Run Quadratic Sieve
    factor1, factor2 = quadratic_sieve(N)
