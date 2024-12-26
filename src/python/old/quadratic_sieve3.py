import numpy as np
import random
import logging
import time
from math import sqrt, ceil, floor, exp, log2, log, isqrt
from line_profiler import LineProfiler

# -------------------------------------------------------------------
# Configure Logging
# -------------------------------------------------------------------
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

prime_log_map = {}

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

def factorise_fast(value, factor_base):
    """
    Factors a number given a factor_base and determines if its smooth over the factor base

    Args:
        value (int): Value to factorise
        factor_base (list): Factor base to factor value over

    Returns:
        list: the factors of the value
        bool: true if the value factorises over the factor base and false otherwise
    """
    factors = []
    if value < 0:
        factors.append(-1)
        value = -value
    for factor in factor_base[1:]:
        while(value % factor == 0):
            factors.append(factor)
            value //= factor
    return sorted(factors), value == 1

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

        mask = np.isin(np.arange(n), pivot_rows)
        relevant_cols = reduced_matrix[:, ones]
        matching_rows = np.any(relevant_cols == 1, axis=1)
        null[mask & matching_rows] = 1

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
    sieve_array = np.ones((n + 1,), dtype=bool)
    sieve_array[0], sieve_array[1] = False, False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve_array[i]:
            sieve_array[i * 2::i] = False
    return np.where(sieve_array)[0].tolist()

def poly(t, n):
    """
    Polynomial used in the Quadratic Sieve: f(t) = t^2 - n.
    
    Args:
        t (int): The variable.
        n (int): The number to factor.
    
    Returns:
        int: The value of the polynomial at t.
    """
    return t*t - n

def find_b(N, reduction=1):
    """
    Typical heuristic to determine the factor base bound B.
    
    Args:
        N (int): The number to factor.
        reduction (int): Number to divide heuristic B value by
    
    Returns:
        int: The bound B for the factor base.
    """
    x = ceil(exp(sqrt(0.5 * log(N) * log(log(N))))) + 1
    return x // reduction

def get_smooth_b(N, B):
    """
    Build the factor base: primes p <= B where (N/p) = 1.
    
    Args:
        N (int): The number to factor.
        B (int): The bound for the factor base.
    
    Returns:
        list: The factor base as a list of primes.
    """
    primes = prime_sieve(B)
    factor_base = [-1, 2]
    prime_log_map[2] = 1
    for p in primes[1:]:  # Skip 2 and check for quadratic residues
        if legendre(N, p) == 1:
            factor_base.append(p)
            prime_log_map[p] = round(log2(p))
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
    logger.info("Factor base size: %d", len(fb))
    return fb

# -------------------------------------------------------------------
# STEP 3: Sieve Phase - Find Potential Smooth Values
# -------------------------------------------------------------------

def sieve_interval(N, factor_base, I_multiplier=14000):
    base = floor(sqrt(N))
    I = len(factor_base) * I_multiplier
    half_I = I // 2

    x_values = [base + x for x in range(-half_I, half_I)]
    sieve_values = [x * x - N for x in x_values]
    sieve_logs = np.zeros(I, dtype=np.float64)

    for p in factor_base:
        if p < 20:
            continue
        root1, root2 = tonelli_shanks(N, p)

        a = (root1 - base + half_I) % p
        b = (root2 - base + half_I) % p

        for r in [a, b]:
            if r < 0:
                continue
            sieve_logs[r::p] += prime_log_map[p]

    return base, I, sieve_logs, sieve_values, x_values

# -------------------------------------------------------------------
# STEP 4: Build Exponent Matrix from Smooth Values
# -------------------------------------------------------------------
def build_exponent_matrix(N, base, I, sieve_logs, sieve_values, factor_base, x_values, B, T=1):
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
    fb_len = len(factor_base)
    zero_row = [0] * fb_len
    error = 17.5


    misses = 0
    hits = 0
    for i in range(I):
        threshold = log2(abs(sieve_values[i])) - error
        if sieve_logs[i] > threshold:
            if len(relations) == fb_len + T:
                break

            value = sieve_values[i]

            local_factors, factored = factorise_fast(value, factor_base)
            if not factored:
                misses += 1
                continue
            hits += 1
            row = zero_row.copy()

            counts = {}
            for fac in local_factors:
                counts[fac] = counts.get(fac, 0) + 1

            for idx, prime in enumerate(factor_base):
                row[idx] = counts.get(prime, 0) % 2

            matrix.append(row)
            relations.append(x_values[i])
            factorizations.append(local_factors)

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
                prod_left *= relations[idx]
                for fac in factorizations[idx]:
                    prod_right *= fac

        sqrt_right = isqrt(prod_right)
        prod_left = prod_left % N
        sqrt_right = sqrt_right % N
        factor_candidate = gcd(N, prod_left - sqrt_right)
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
    logger.info("========== Quadratic Sieve V3 Start ==========")
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
    base, I, sieve_logs, sieve_vals, x_values = sieve_interval(N, factor_base)
    step_end = time.time()
    logger.info("Step 3 (Sieve Interval) took %.3f seconds", step_end - step_start)

    # Step 4: Build Exponent Matrix
    step_start = time.time()
    matrix, relations, factorizations = build_exponent_matrix(N, base, I, sieve_logs, sieve_vals, factor_base, x_values, B, T=1)
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
    # Example composite numbers
    N = 87463
    #N = 87463
    #N = 841921111621030451922256098390257311

    # Run Quadratic Sieve
    factor1, factor2 = quadratic_sieve(N)
