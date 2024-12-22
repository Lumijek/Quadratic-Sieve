import numpy as np
import random
import logging
from math import ceil, floor, sqrt, exp, log, isqrt
import galois

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
    Initialize the known primes up to 'limit' using 'is_prime' test.
    This helps optimize primality checks for smaller numbers.
    """
    global _known_primes
    _known_primes += [x for x in range(5, limit, 2) if is_prime(x)]
    logger.info("Initialized _known_primes up to %d. Total known primes: %d", limit, len(_known_primes))

# -------------------------------------------------------------------
# Number Theory Utils
# -------------------------------------------------------------------
def gcd(a, b):
    """
    Compute the GCD of two integers a and b using Euclid's Algorithm.
    """
    a, b = abs(a), abs(b)
    while a:
        a, b = b % a, a
    return b

def legendre(n, p):
    """
    Compute the Legendre symbol (n/p).
    The Legendre symbol is defined for p an odd prime.
    Returns:
       1  if n is a quadratic residue mod p
       -1 if n is a non-quadratic residue mod p
        0 if n ≡ 0 (mod p)
    """
    val = pow(n, (p - 1) // 2, p)
    return val - p if val > 1 else val

def jacobi(a, m):
    """
    Compute the Jacobi symbol (a/m). 
    Returns:
       1  if a is a QR modulo each prime factor of m
       -1 if a is a non-QR for at least one prime factor
        0 if a and m are not coprime
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
    """
    if n in _known_primes:
        return True
    if any((n % p) == 0 for p in _known_primes) or n in (0, 1):
        return n in _known_primes  # ensures 2 or 3 recognized as prime, others as not prime

    d, s = n - 1, 0
    while d % 2 == 0:
        d >>= 1
        s += 1

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

    return not any(_try_composite(a, d, n, s) 
                   for a in _known_primes[:_precision_for_huge_n])

# -------------------------------------------------------------------
# Factorization (Brent's Method)
# -------------------------------------------------------------------
def brent(N):
    """
    Brent's factorization algorithm (variation of Pollard's Rho).
    Returns a non-trivial factor of N.
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
            y = (y*y % N + c) % N
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y*y % N + c) % N
                q = q * abs(x - y) % N
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
    """
    rem = n
    while True:
        if is_prime(rem):
            factors.append(rem)
            break

        f = brent(rem)
        # if the factor is the same as remainder, try again
        while f == rem:
            f = brent(rem)

        if f and f < rem:
            if is_prime(f):
                factors.append(f)
                rem //= f
            else:
                rem_f = factorise(f, factors)
                rem = (rem // f) * rem_f
                # remove rem_f if it was appended
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
        i2 = A * pow(D, m, p) % p
        i3 = pow(i2, i1, p)
        if i3 == p - 1:
            m += pow(2, i)
    x = (pow(a, (t+1)//2, p) * pow(D, m//2, p)) % p
    return x, p - x

# -------------------------------------------------------------------
# Prime Sieve
# -------------------------------------------------------------------
def prime_sieve(n):
    """
    Return a list of primes up to 'n' using the Sieve of Eratosthenes.
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
    Polynomial for Quadratic Sieve: f(t) = t^2 - n.
    """
    return pow(t, 2) - n

def find_b(N):
    """
    Typical heuristic for factor base bound B.
    """
    return ceil(exp(sqrt(0.5 * log(N) * log(log(N))))) + 1

def get_smooth_b(N, B):
    """
    Build the factor base: primes p <= B where (N/p) = 1.
    """
    primes = prime_sieve(B)
    factor_base = [2]  # Always include 2
    for p in primes[1:]:  # skip 2
        if legendre(N, p) == 1:
            factor_base.append(p)
    return factor_base

# -------------------------------------------------------------------
# STEP 1: Decide Bound B
# -------------------------------------------------------------------
def decide_bound(N, B=None):
    """
    Decide on the bound B if none is provided.
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
    """
    fb = get_smooth_b(N, B)
    logger.info("Factor base size: %d", len(fb))
    return fb

# -------------------------------------------------------------------
# STEP 3: Sieve Phase - Find Potential Smooth Values
# -------------------------------------------------------------------
def sieve_interval(N, factor_base, I_multiplier=182):
    """
    Sieve to find potential smooth values in the interval [base, base+I).
    Returns:
       - The offset base (floor(sqrt(N)) + 1)
       - The length of the interval
       - The array of partially factored values
    """
    # Interval size is factor_base_size * I_multiplier
    I = len(factor_base) * I_multiplier
    base = floor(sqrt(N)) + 1

    # Evaluate polynomial t^2 - N over the interval
    sieve_values = [poly(base + x, N) for x in range(I)]

    # Remove powers of 2 from each value
    for i in range(I):
        while sieve_values[i] % 2 == 0:
            sieve_values[i] //= 2

    # For each prime in the factor base (except 2), remove its powers
    for p in factor_base[1:]:
        # Solve x^2 ≡ N (mod p) => x = ± sqrt(N mod p)
        # Then map it to offsets in [0..I)
        root1, root2 = tonelli_shanks(N, p)
        # Shift by -base, mod p
        a = (root1 - base) % p
        b = (root2 - base) % p

        # Remove powers of p from valid offsets
        for r in [a, b]:
            while r < I:
                while sieve_values[r] % p == 0:
                    sieve_values[r] //= p
                r += p

    return base, I, sieve_values

# -------------------------------------------------------------------
# STEP 4: Build Exponent Matrix from Smooth Values
# -------------------------------------------------------------------
def build_exponent_matrix(N, base, I, sieve_values, factor_base, T=1):
    """
    For each index i in [0..I), if the sieve_values[i] is 1,
    we factor poly(base + i, N) fully to build a row in the exponent matrix.
    """
    matrix = []
    relations = []
    factorizations = []
    fb_len = len(factor_base)
    zero_row = [0] * fb_len

    for i_offset in range(I):
        if sieve_values[i_offset] == 1:
            if len(relations) == len(factor_base) + T:
                break
            # This value is fully divisible by the factor base
            row = zero_row.copy()
            value = poly(base + i_offset, N)

            # Factor completely (to account for multiplicities)
            local_factors = []
            factorise(value, local_factors)
            local_factors.sort()

            # Count each prime factor mod 2
            counts = {}
            for fac in local_factors:
                counts[fac] = counts.get(fac, 0) + 1

            for idx, prime in enumerate(factor_base):
                row[idx] = counts.get(prime, 0) % 2

            matrix.append(row)
            relations.append(base + i_offset)
            factorizations.append(local_factors)

    logger.info("Number of smooth relations: %d", len(relations))
    return matrix, relations, factorizations

# -------------------------------------------------------------------
# STEP 5: Solve for Dependencies in GF(2)
# -------------------------------------------------------------------
def solve_dependencies(matrix):
    """
    Perform linear algebra over GF(2) to find the left null space.
    """
    logger.info("Solving linear system in GF(2).")
    gf = galois.GF(2)
    A = gf(np.array(matrix))
    # Vectors 'v' in left null space satisfy vA = 0
    return A.left_null_space()

# -------------------------------------------------------------------
# STEP 6: Attempt to Extract Factors
# -------------------------------------------------------------------
def extract_factors(N, relations, factorizations, dep_vectors):
    """
    Use the dependency vectors (solutions in GF(2)) to attempt factor extraction.
    Returns (p, q) if a non-trivial factor pair is found, otherwise (0, 0).
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
        factor_candidate = gcd(N, prod_left - sqrt_right)
        if factor_candidate not in (1, N):
            other_factor = N // factor_candidate
            logger.info("Found factors: %d, %d", factor_candidate, other_factor)
            return factor_candidate, other_factor

    return 0, 0

# -------------------------------------------------------------------
# Main Quadratic Sieve Function
# -------------------------------------------------------------------
def quadratic_sieve(N, B=None):
    """
    Perform the Quadratic Sieve on N.
    Splits the entire process into smaller steps.
    """
    logger.info("========== Quadratic Sieve Start ==========")
    logger.info("Factoring N = %d", N)

    # Step 1: Decide Bound
    B = decide_bound(N, B)

    # Step 2: Build Factor Base
    factor_base = build_factor_base(N, B)

    # Step 3: Sieve Phase
    base, I, sieve_vals = sieve_interval(N, factor_base)

    # Step 4: Build Exponent Matrix
    matrix, relations, factorizations = build_exponent_matrix(
        N, base, I, sieve_vals, factor_base
    )

    if len(matrix) < len(factor_base):
        logger.warning("Not enough smooth relations found. Try increasing the sieve interval.")
        return 0, 0

    # Step 5: Solve for Dependencies (GF(2))
    dep_vectors = solve_dependencies(matrix)

    # Step 6: Attempt to Extract Factors
    f1, f2 = extract_factors(N, relations, factorizations, dep_vectors)
    if f1 and f2:
        logger.info("Quadratic Sieve successful: %d * %d = %d", f1, f2, N)
    else:
        logger.warning("No non-trivial factors found with the current settings.")
    logger.info("========== Quadratic Sieve End ==========")

    return f1, f2

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize small primes list
    init_known_primes(limit=1000)

    # Example composite number
    N = 41126566532996951593624199 * 2697660919
    N = 489676279211111 * 230114213738729
    # Run Quadratic Sieve
    factor1, factor2 = quadratic_sieve(N)
