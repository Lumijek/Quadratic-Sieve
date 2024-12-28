import numpy as np
import random
import logging
import time
from collections import deque
from math import sqrt, ceil, floor, exp, log2, log, isqrt
import gc


class QuadraticSieve:
    def __init__(
        self,
        I_multiplier: int = 10000,
        reduction: int = 1,
        T: int = 1
    ):
        """
        Initialize the Quadratic Sieve with logging configuration and hyperparameters.

        Args:
            I_multiplier (int): Multiplier to determine sieve interval size (default: 40000).
            reduction (int): Reduction factor for the bound B (default: 1).
            T (int): Additional threshold for relations (default: 1).
        """
        # Configure logging with the specified format and level
        logging.basicConfig(
            format='[%(levelname)s] %(asctime)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.prime_log_map = {}
        
        # Store hyperparameters
        self.I_multiplier = I_multiplier
        self.reduction = reduction
        self.T = T

    @staticmethod
    def gcd(a, b):
        """Compute GCD of two integers using Euclid's Algorithm."""
        a, b = abs(a), abs(b)
        while a:
            a, b = b % a, a
        return b

    @staticmethod
    def legendre(n, p):
        """Compute the Legendre symbol (n/p)."""
        val = pow(n, (p - 1) // 2, p)
        return val - p if val > 1 else val

    @staticmethod
    def jacobi(a, m):
        """Compute the Jacobi symbol (a/m)."""
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

    def factorise_fast(self, value, factor_base):
        """Factor a number over the given factor base.

        Args:
            value: The integer to factorize.
            factor_base: A list of primes forming the factor base.

        Returns:
            A tuple containing a sorted list of factors and the remaining unfactored value.
        """
        factors = []
        if value < 0:
            factors.append(-1)
            value = -value
        for factor in factor_base[1:]:
            while value % factor == 0:
                factors.append(factor)
                value //= factor
        return sorted(factors), value

    @staticmethod
    def tonelli_shanks(a, p):
        """Solve x^2 ≡ a (mod p) for x using the Tonelli-Shanks algorithm.

        Args:
            a: The quadratic residue.
            p: The prime modulus.

        Returns:
            A tuple containing the two square roots modulo p.
        """
        a %= p
        if p % 8 in [3, 7]:
            x = pow(a, (p + 1) // 4, p)
            return x, p - x

        if p % 8 == 5:
            x = pow(a, (p + 3) // 8, p)
            if pow(x, 2, p) != a % p:
                x = (x * pow(2, (p - 1) // 4, p)) % p
            return x, p - x

        # Find a quadratic non-residue d
        d = 2
        symb = 0
        while symb != -1:
            symb = QuadraticSieve.jacobi(d, p)
            d += 1
        d -= 1

        # Factor p-1 as Q * 2^S
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
        x = (pow(a, (t + 1) // 2, p) * pow(D, m // 2, p)) % p
        return x, p - x

    @staticmethod
    def gauss_elim(x):
        """Perform Gaussian elimination on a binary matrix over GF(2).

        Args:
            x: A NumPy array representing the binary matrix.

        Returns:
            A tuple containing the reduced matrix and a sorted list of pivot column indices.
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

    @staticmethod
    def find_null_space_GF2(reduced_matrix, pivot_rows):
        """Find null space vectors of the reduced binary matrix over GF(2).

        Args:
            reduced_matrix: The reduced binary matrix.
            pivot_rows: A list of pivot column indices.

        Returns:
            A NumPy array containing null space vectors.
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

        return np.asarray(nulls, dtype=np.int8)

    @staticmethod
    def prime_sieve(n):
        """Return list of primes up to n using Sieve of Eratosthenes.

        Args:
            n: The upper limit for prime generation.

        Returns:
            A list of prime numbers up to n.
        """
        sieve_array = np.ones((n + 1,), dtype=bool)
        sieve_array[0], sieve_array[1] = False, False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve_array[i]:
                sieve_array[i * 2::i] = False
        return np.where(sieve_array)[0].tolist()

    def find_b(self, N, reduction=None):
        """Determine the factor base bound B.

        Args:
            N: The integer to factorize.
            reduction: Reduction factor for the bound (default: self.reduction).

        Returns:
            The computed bound B.
        """
        if reduction is None:
            reduction = self.reduction
        x = ceil(exp(sqrt(0.5 * log(N) * log(log(N))))) + 1
        return x // reduction

    def get_smooth_b(self, N, B):
        """Build the factor base of primes p ≤ B where (N/p) = 1.

        Args:
            N: The integer to factorize.
            B: The bound for the factor base.

        Returns:
            A list representing the factor base.
        """
        primes = self.prime_sieve(B)
        factor_base = [-1, 2]
        self.prime_log_map[2] = 1
        for p in primes[1:]:
            if self.legendre(N, p) == 1:
                factor_base.append(p)
                self.prime_log_map[p] = round(log2(p))
        return factor_base

    def decide_bound(self, N, B=None):
        """Decide on bound B using heuristic if none provided.

        Args:
            N: The integer to factorize.
            B: Optional bound for the factor base.

        Returns:
            The decided bound B.
        """
        if B is None:
            B = self.find_b(N)
        self.logger.info("Using B = %d", B)
        return B

    def build_factor_base(self, N, B):
        """Build factor base for Quadratic Sieve.

        Args:
            N: The integer to factorize.
            B: The bound for the factor base.

        Returns:
            The constructed factor base.
        """
        fb = self.get_smooth_b(N, B)
        self.logger.info("Factor base size: %d", len(fb))
        return fb

    def sieve_interval(self, N, factor_base, I_multiplier=None):
        """Perform sieving over an interval.

        Args:
            N: The integer to factorize.
            factor_base: The factor base.
            I_multiplier: Multiplier to determine sieve interval size (default: self.I_multiplier).

        Returns:
            A tuple containing base, interval size I, sieve logs, sieve values, and x values.
        """
        if I_multiplier is None:
            I_multiplier = self.I_multiplier
        base = floor(sqrt(N))
        I = len(factor_base) * I_multiplier
        half_I = I // 2

        x_values = [base + x for x in range(-half_I, half_I)]
        sieve_values = [x * x - N for x in x_values]
        sieve_logs = np.zeros(I, dtype=np.float64)

        for p in factor_base:
            if p < 20:
                continue
            try:
                root1, root2 = self.tonelli_shanks(N, p)
            except ValueError:
                continue  # Skip primes where no roots exist

            a = (root1 - base + half_I) % p
            b = (root2 - base + half_I) % p

            for r in [a, b]:
                sieve_logs[r::p] += self.prime_log_map.get(p, log2(p))

        return base, I, sieve_logs, sieve_values, x_values

    def build_exponent_matrix(
        self,
        N,
        base,
        I,
        sieve_logs,
        sieve_values,
        factor_base,
        x_values,
        B,
        T=None
    ):
        """Build exponent matrix from smooth values.

        Args:
            N: The integer to factorize.
            base: The base value from sieving.
            I: The interval size.
            sieve_logs: Logarithmic sieve values.
            sieve_values: List of sieve values (x^2 - N).
            factor_base: The factor base.
            x_values: Corresponding x values.
            B: The bound for the factor base.
            T: Additional threshold for relations (default: self.T).

        Returns:
            A tuple containing the exponent matrix, relations, and their factorizations.
        """
        if T is None:
            T = self.T
        matrix = []
        relations = []
        factorizations = []
        fb_len = len(factor_base)
        zero_row = [0] * fb_len
        error = 20

        large_prime_bound = B * 128
        partials = {}
        lp_found = 0
        count = 0

        for i in range(I):
            count += 1
            if sieve_values[i] == 0:
                continue  # Avoid log2(0) error
            threshold = log2(abs(sieve_values[i])) - error
            if sieve_logs[i] > threshold:
                if len(relations) == fb_len + T:
                    break
                mark = False
                relation = x_values[i]
                local_factors, value = self.factorise_fast(sieve_values[i], factor_base)

                if value != 1 and value < large_prime_bound:
                    if value not in partials:
                        partials[value] = (x_values[i], local_factors)
                        continue
                    else:
                        lp_found += 1
                        relation = relation * partials[value][0]
                        local_factors += partials[value][1]
                        mark = True
                elif value != 1:
                    continue

                row = zero_row.copy()
                counts = {}
                for fac in local_factors:
                    counts[fac] = counts.get(fac, 0) + 1

                for idx, prime in enumerate(factor_base):
                    row[idx] = counts.get(prime, 0) % 2

                if mark:
                    local_factors.append(value * value)

                matrix.append(row)
                relations.append(relation)
                factorizations.append(local_factors)

        self.logger.info("Number of relations using partial relations: %d", lp_found)
        self.logger.info("Number of relations using sieving: %d", len(relations) - lp_found)
        self.logger.info("Number of smooth relations: %d", len(relations))
        return matrix, relations, factorizations

    def solve_dependencies(self, matrix):
        """Solve for dependencies in GF(2).

        Args:
            matrix: The exponent matrix.

        Returns:
            Null space vectors representing dependencies.
        """
        self.logger.info("Solving linear system in GF(2).")
        matrix = np.array(matrix).T
        reduced_matrix, marks = self.gauss_elim(matrix)
        null_basis = self.find_null_space_GF2(reduced_matrix.T, marks)
        return null_basis

    def extract_factors(self, N, relations, factorizations, dep_vectors):
        """Extract factors using dependency vectors.

        Args:
            N: The integer to factorize.
            relations: List of relations (smooth numbers).
            factorizations: List of factorizations corresponding to relations.
            dep_vectors: Dependency vectors from the null space.

        Returns:
            A tuple containing two non-trivial factors of N or (0, 0) if unsuccessful.
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
            factor_candidate = self.gcd(N, prod_left - sqrt_right)

            if factor_candidate not in (1, N):
                other_factor = N // factor_candidate
                self.logger.info("Found factors: %d, %d", factor_candidate, other_factor)
                return factor_candidate, other_factor

        return 0, 0

    def factor(self, N, B=None):
        """Main factorization method using the Quadratic Sieve algorithm.

        Args:
            N: The integer to factorize.
            B: Optional bound for the factor base.

        Returns:
            A tuple containing two non-trivial factors of N or (0, 0) if unsuccessful.
        """
        overall_start = time.time()
        self.logger.info("========== Quadratic Sieve V4 Start ==========")
        self.logger.info("Factoring N = %d", N)

        # Step 1: Decide Bound
        step_start = time.time()
        B = self.decide_bound(N, B)
        step_end = time.time()
        self.logger.info("Step 1 (Decide Bound) took %.3f seconds", step_end - step_start)

        # Step 2: Build Factor Base
        step_start = time.time()
        factor_base = self.build_factor_base(N, B)
        step_end = time.time()
        self.logger.info("Step 2 (Build Factor Base) took %.3f seconds", step_end - step_start)

        # Step 3: Sieve Phase
        step_start = time.time()
        base, I, sieve_logs, sieve_vals, x_values = self.sieve_interval(N, factor_base)
        step_end = time.time()
        self.logger.info("Step 3 (Sieve Interval) took %.3f seconds", step_end - step_start)

        # Step 4: Build Exponent Matrix
        step_start = time.time()
        matrix, relations, factorizations = self.build_exponent_matrix(
            N, base, I, sieve_logs, sieve_vals, factor_base, x_values, B, T=self.T
        )
        step_end = time.time()
        self.logger.info("Step 4 (Build Exponent Matrix) took %.3f seconds", step_end - step_start)

        if len(matrix) < len(factor_base) + 1:
            self.logger.warning("Not enough smooth relations found. Try increasing the sieve interval.")
            return 0, 0

        # Step 5: Solve for Dependencies
        step_start = time.time()
        dep_vectors = self.solve_dependencies(np.array(matrix))
        step_end = time.time()
        self.logger.info("Step 5 (Solve Dependencies) took %.3f seconds", step_end - step_start)

        # Step 6: Extract Factors
        step_start = time.time()
        f1, f2 = self.extract_factors(N, relations, factorizations, dep_vectors)
        step_end = time.time()
        self.logger.info("Step 6 (Extract Factors) took %.3f seconds", step_end - step_start)

        if f1 and f2:
            self.logger.info("Quadratic Sieve successful: %d * %d = %d", f1, f2, N)
        else:
            self.logger.warning("No non-trivial factors found with the current settings.")

        overall_end = time.time()
        self.logger.info("Total time for Quadratic Sieve: %.3f seconds", overall_end - overall_start)
        self.logger.info("========== Quadratic Sieve End ==========")

        return f1, f2


if __name__ == '__main__':
    # Example usage
    N = 97245170828229363259 * 49966345331749027373
    gc.disable()

    # Create QuadraticSieve instance with default hyperparameters
    qs = QuadraticSieve(I_multiplier=40000, reduction=24, T=1)

    # Alternatively, pass custom hyperparameters
    # qs = QuadraticSieve(I_multiplier=50000, reduction=2, T=2)

    factor1, factor2 = qs.factor(N)

    # Output the results
    if factor1 and factor2:
        print(f"Factors of N: {factor1} * {factor2} = {N}")
    else:
        print("Failed to factorize N with the current settings.")
