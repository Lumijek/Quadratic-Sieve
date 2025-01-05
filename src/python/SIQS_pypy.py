import logging
import time
import math
from math import sqrt, ceil, floor, exp, log2, log, isqrt
from rich.live import Live
from rich.table import Table
from rich.console import Console
import random
import sys

LOWER_BOUND_SIQS = 1000
UPPER_BOUND_SIQS = 4000
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO
)
def get_gray_code(n):
    gray = [0] * (1 << (n - 1))
    gray[0] = (0, 0)
    for i in range(1, 1 << (n - 1)):
        v = 1
        j = i
        while (j & 1) == 0:
            v += 1
            j >>= 1
        tmp = i + ((1 << v) - 1)
        tmp >>= v
        if (tmp & 1) == 1:
            gray[i] = (v - 1, -1)
        else:
            gray[i] = (v - 1, 1)
    return gray

MULT_LIST = [
     1,   2,   3,   5,   7,   9,  10,  11,  13,  14,
    15,  17,  19,  21,  23,  25,  29,  31,
    33,  35,  37,  39,  41,  43,  45,   
    47,  49,  51,  53,  55,  57,  59,  61,  63,
    65,  67,  69,  71,  73,  75,  77,  79,  83,
    85,  87,  89,  91,  93,  95,  97, 101, 103, 105, 107,
   109, 111, 113, 115, 119, 121, 123, 127, 129, 131, 133,
   137, 139, 141, 143, 145, 147, 149, 151, 155, 157, 159,
   161, 163, 165, 167, 173, 177, 179, 181, 183, 185, 187,
   191, 193, 195, 197, 199, 201, 203, 205, 209, 211, 213,
   215, 217, 219, 223, 227, 229, 231, 233, 235, 237, 239,
   241, 249, 251, 253, 255
]

def create_table(relations, target_relations, num_poly, start_time):
    end = time.time()
    elapsed = end - start_time

    relations_per_second = len(relations) / elapsed if elapsed > 0 else 0
    poly_per_second = num_poly / elapsed if elapsed > 0 else 0
    percent = (len(relations) / target_relations) * 100 if target_relations > 0 else 0
    percent_per_second = percent / elapsed if elapsed > 0 else 0
    remaining_percent = 100.0 - percent
    seconds = int(remaining_percent / percent_per_second) if percent_per_second > 0 else 0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    table = Table(title="Processing Status")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Relations per second", f"{relations_per_second:,.2f}")
    table.add_row("Poly per second", f"{poly_per_second:,.2f}")
    table.add_row("Percent", f"{percent:,.2f}%")
    table.add_row("Percent per second", f"{percent_per_second:,.4f}%")
    table.add_row("Estimated Time", f"{h:d}:{m:02d}:{s:02d}")

    return table

class QuadraticSieve:
    def __init__(self, M, B=None, T=2, prime_limit=20, eps=30, lp_multiplier=20, multiplier=None):
        self.logger = logging.getLogger(__name__)
        self.prime_log_map = {}
        self.root_map = {}
        
        self.M = M
        self.B = B
        self.T = T
        self.prime_limit = prime_limit
        self.eps = eps
        self.lp_multiplier = lp_multiplier
        self.multiplier = multiplier

        self.console = Console()

        print(f"B: {B}")
        print(f"M: {M}")
        print(f"prime_limit: {prime_limit}")
        print(f"eps: {eps}")
        print(f"lp_multiplier: {lp_multiplier}")

    @staticmethod
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while a:
            a, b = b % a, a
        return b

    @staticmethod
    def legendre(n, p):
        val = pow(n, (p - 1) // 2, p)
        return val - p if val > 1 else val

    @staticmethod
    def jacobi(a, m):
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

    @staticmethod
    def modinv(n, p):
        n = n % p
        x, u = 0, 1
        while n:
            x, u = u, x - (p // n) * u
            p, n = n, p % n
        return x

    def factorise_fast(self, value, factor_base):
        factors = set()
        if value < 0:
            factors ^= {-1}
            value = -value
        for factor in factor_base[1:]:
            while value % factor == 0:
                factors ^= {factor}
                value //= factor
        return factors, value

    @staticmethod
    def tonelli_shanks(a, p):
        a %= p
        if p % 8 in [3, 7]:
            x = pow(a, (p + 1) // 4, p)
            return x, p - x

        if p % 8 == 5:
            x = pow(a, (p + 3) // 8, p)
            if pow(x, 2, p) != a % p:
                x = (x * pow(2, (p - 1) // 4, p)) % p
            return x, p - x

        d = 2
        symb = 0
        while symb != -1:
            symb = QuadraticSieve.jacobi(d, p)
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
        x = (pow(a, (t + 1) // 2, p) * pow(D, m // 2, p)) % p
        return x, p - x

    @staticmethod
    def prime_sieve(n):
        sieve = [True] * (n + 1)
        sieve[0], sieve[1] = False, False

        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i * 2, n + 1, i):
                    sieve[j] = False

        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def find_b(self, N):
        x = ceil(exp(0.5 * sqrt(log(N) * log(log(N)))))
        return x

    def choose_multiplier(self, N, B):
        prime_list = self.prime_sieve(B)
        if self.multiplier is not None:
            self.logger.info("Using multiplier k = %d", self.multiplier)
            return prime_list
        NUM_TEST_PRIMES = 300
        LN2 = math.log(2)
        num_primes = min(len(prime_list), NUM_TEST_PRIMES)
        log2n = math.log(N)
        scores = [0.0 for _ in MULT_LIST]
        num_multipliers = 0

        for i, curr_mult in enumerate(MULT_LIST):
            knmod8 = (curr_mult * (N % 8)) % 8
            logmult = math.log(curr_mult)
            scores[i] = 0.5 * logmult

            if knmod8 == 1:
                scores[i] -= 2 * LN2
            elif knmod8 == 5:
                scores[i] -= LN2
            elif knmod8 in (3, 7):
                scores[i] -= 0.5 * LN2

            num_multipliers += 1

        for i in range(1, num_primes):
            prime = prime_list[i]
            contrib = math.log(prime) / (prime - 1)
            modp = N % prime

            for j in range(num_multipliers):
                curr_mult = MULT_LIST[j]
                knmodp = (modp * curr_mult) % prime

                if knmodp == 0 or self.legendre(knmodp, prime) == 1:
                    if knmodp == 0:
                        scores[j] -= contrib
                    else:
                        scores[j] -= 2 * contrib

        best_score = float('inf')
        best_mult = 1

        for i in range(num_multipliers):
            if scores[i] < best_score:
                best_score = scores[i]
                best_mult = MULT_LIST[i]

        self.multiplier = best_mult
        self.logger.info("Using multiplier k = %d", best_mult)
        return prime_list

    def get_smooth_b(self, N, B, prime_list):
        factor_base = [-1, 2]
        self.prime_log_map[2] = 1
        for p in prime_list[1:]:
            if self.legendre(N, p) == 1:
                factor_base.append(p)
                self.prime_log_map[p] = round(log2(p))
                self.root_map[p] = self.tonelli_shanks(N, p)

        return factor_base

    def decide_bound(self, N, B=None):
        if B is None:
            B = self.find_b(N)
        self.B = B
        self.logger.info("Using B = %d", B)
        return B

    def build_factor_base(self, N, B, prime_list):
        fb = self.get_smooth_b(N, B, prime_list)
        self.logger.info("Factor base size: %d", len(fb))
        return fb

    def new_poly_a(self, factor_base, N, M, poly_a_list):
        small_B = 1024
        lower_polypool_index = 2
        upper_polypool_index = small_B - 1
        poly_low_found = False
        for i in range(small_B):
            if factor_base[i] > LOWER_BOUND_SIQS and not poly_low_found:
                lower_polypool_index = i
                poly_low_found = True
            if factor_base[i] > UPPER_BOUND_SIQS:
                upper_polypool_index = i - 1
                break
        # Compute target_a and bit threshold
        target_a = int(math.sqrt(2 * N) / M)
        target_mul = 0.9
        target_bits = int(target_a.bit_length() * target_mul)
        too_close = 10
        close_range = 5
        min_ratio = LOWER_BOUND_SIQS

        while True:
            poly_a = 1
            afact = []
            qli = []
            while True:
                found_a_factor = False
                while(found_a_factor == False):
                    randindex = random.randint(lower_polypool_index, upper_polypool_index)
                    potential_a_factor = factor_base[randindex]
                    found_a_factor = True
                    if potential_a_factor in afact:
                        found_a_factor = False

                poly_a = poly_a * potential_a_factor
                afact.append(potential_a_factor)
                qli.append(randindex)

                j = target_a.bit_length() - poly_a.bit_length()
                if j < too_close:
                    poly_a = 1
                    s = 0
                    afact = []
                    qli = []
                    continue
                elif j < (too_close + close_range):
                    break

            a1 = target_a // poly_a
            if a1 < min_ratio:
                continue

            mindiff = 100000000000000000
            randindex = 0

            for i in range(small_B):
                if abs(a1 - factor_base[i]) < mindiff:
                    mindiff = abs(a1 - factor_base[i])
                    randindex = i

            found_a_factor = False
            while not found_a_factor:
                potential_a_factor = factor_base[randindex]
                found_a_factor = True
                if potential_a_factor in afact:
                    found_a_factor = False
                if not found_a_factor:
                    randindex += 1

            if randindex > small_B:
                continue

            poly_a = poly_a * factor_base[randindex]
            afact.append(factor_base[randindex])
            qli.append(randindex)

            diff_bits = (target_a - poly_a).bit_length()

            if diff_bits < target_bits:
                if poly_a in poly_a_list:
                    if target_bits > 1000:
                        print("SOMETHING WENT WRONG")
                        sys.exit()
                    target_bits += 1
                    continue
                else:
                    break

        poly_a_list.append(poly_a)
        return poly_a, sorted(qli), set(afact)

    def generate_first_polynomial(self, factor_base, N, M, poly_a_list):
        a, qli, factors_a = self.new_poly_a(factor_base, N, M, poly_a_list)
        s = len(qli)
        B = []
        for l in range(s):
            p = factor_base[qli[l]]
            r1 = self.root_map[p][0]
            aq = a // p
            invaq = self.modinv(aq, p)
            gamma = r1 * invaq % p
            if gamma > p // 2:
                gamma = p - gamma
            B.append(aq * gamma)
        b = sum(B) % a
        c = (b * b - N) // a
        soln_map = {}
        Bainv = {}
        for p in factor_base:
            Bainv[p] = []
            if a % p == 0 or p == 2:
                continue

            ainv = self.modinv(a, p)
            # store bainv
            for j in range(s):
                Bainv[p].append((2 * B[j] * ainv) % p)

            # store roots
            r1, r2 = self.root_map[p]
            r1 = ((r1 - b) * ainv) % p
            r2 = ((r2 - b) * ainv) % p
            soln_map[p] = [r1, r2]

        return a, b, c, B, Bainv, soln_map, s, factors_a


    def sieve(self, N, B, factor_base, M):
        # ------------------------------------------------
        # 1) TIMING
        # ------------------------------------------------
        start = time.time()

        # ------------------------------------------------
        # 2) FACTOR BASE & RELATED
        # ------------------------------------------------
        fb_len = len(factor_base)
        fb_map = {val: i for i, val in enumerate(factor_base)}
        target_relations = fb_len + self.T
        large_prime_bound = B * self.lp_multiplier

        # ------------------------------------------------
        # 3) THRESHOLD & MISC
        # ------------------------------------------------
        threshold = int(math.log2(M * math.sqrt(N)) - self.eps)
        lp_found = 0
        ind = 1

        # ------------------------------------------------
        # 4) STORAGE FOR RESULTS & PARTIALS
        # ------------------------------------------------
        matrix = [0] * fb_len
        relations = []
        roots = []
        partials = {}

        # ------------------------------------------------
        # 5) POLYNOMIAL CONTROLS
        # ------------------------------------------------
        num_poly = 0
        interval_size = 2 * M + 1
        grays = get_gray_code(20)
        poly_a_list = []
        poly_ind = 0

        # ------------------------------------------------
        # 6) SIEVE ARRAY & TEMP VARS
        # ------------------------------------------------
        sieve_values = [0] * interval_size
        r1 = 0
        r2 = 0

        # ------------------------------------------------
        # 7) HELPER FUNCTION
        # ------------------------------------------------
        def process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a):
            nonlocal ind
            val = sieve_values[x]
            sieve_values[x] = 0
            lpf = 0
            if val > threshold:
                xval = x - M
                relation = a * xval + b
                poly_val = a * xval * xval + 2 * b * xval + c

                local_factors, value = self.factorise_fast(poly_val, factor_base)
                local_factors ^= factors_a


                if value != 1:
                    if value < large_prime_bound:
                        if value in partials:
                            rel, lf, pv = partials[value]
                            relation *= rel
                            local_factors ^= lf
                            poly_val *= pv
                            lpf = 1
                        else:
                            partials[value] = (relation, local_factors, poly_val * a)
                            return 0
                    else:
                        return 0

                for fac in local_factors:
                    idx = fb_map[fac]
                    matrix[idx] |= ind
                ind = ind + ind
                relations.append(relation)
                roots.append(poly_val * a)
            return lpf

        with Live(console=self.console) as live:
            while len(relations) < target_relations:
                if num_poly % 10 == 0:
                    live.update(create_table(relations, target_relations, num_poly, start))

                if poly_ind == 0:
                    a, b, c, B, Bainv, soln_map, s, factors_a = self.generate_first_polynomial(factor_base, N, M, poly_a_list)
                    end = 1 << (s - 1)
                    poly_ind += 1
                else:
                    v, e = grays[poly_ind]
                    b = (b + 2 * e * B[v])
                    c = (b * b - N) // a
                    poly_ind += 1
                    if poly_ind == end:
                        poly_ind = 0
                v, e = grays[poly_ind] # v, e for next iteration
                plm = self.prime_log_map
                for p in factor_base:
                    if p < self.prime_limit or a % p == 0:
                        continue
                    log_p = plm[p]
                    r1, r2 = soln_map[p]
                    soln_map[p][0] = (r1 - e * Bainv[p][v]) % p
                    soln_map[p][1] = (r2 - e * Bainv[p][v]) % p

                    amx = r1 + M
                    bmx = r2 + M

                    apx = amx - p
                    bpx = bmx - p
                    k = p

                    while k < M:
                        sieve_values[apx + k] += log_p
                        sieve_values[bpx + k] += log_p
                        sieve_values[amx - k] += log_p
                        sieve_values[bmx - k] += log_p
                        k += p
                num_poly += 1

                x = 0
                while x < 2 * M - 6:
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1
                    lp_found += process_sieve_value(x, sieve_values, partials, relations, roots, a, b, c, factors_a)
                    x += 1

        print(f"\n{num_poly} polynomials sieved")
        print(f"{lp_found} relations from partials")
        print(f"{target_relations - lp_found} normal smooth relations")
        print(f"{target_relations} total relations\n")
        return matrix, relations, roots

    def solve_bits(self, matrix, n):
        self.logger.info("Solving linear system in GF(2).")
        lsmap = {lsb: 1 << lsb for lsb in range(n)}

        # GAUSSIAN ELIMINATION
        m = len(matrix)
        marks = []
        cur = -1
        # m -> number of primes in factor base
        # n -> number of smooth relations
        mark_mask = 0
        for row in matrix:
            if cur % 100 == 0:
                print("", end=f"{cur, m}\r")
            cur += 1
            lsb = (row & -row).bit_length() - 1
            if lsb == -1:
                continue
            marks.append(n - lsb - 1)
            mark_mask |= 1 << lsb
            for i in range(m):
                if matrix[i] & lsmap[lsb] and i != cur:
                    matrix[i] ^= row
        marks.sort()
        # NULL SPACE EXTRACTION
        nulls = []
        free_cols = [col for col in range(n) if col not in marks]
        k = 0
        for col in free_cols:
            shift = n - col - 1
            val = 1 << shift
            fin = val
            for v in matrix:
                if v & val:
                    fin |= v & mark_mask
            nulls.append(fin)
            k += 1
            if k == self.T:
                break
        return nulls

    def extract_factors(self, N, relations, roots, null_space):
        n = len(relations)
        for vector in null_space:
            prod_left = 1
            prod_right = 1
            for idx in range(len(relations)):
                bit = vector & 1
                vector = vector >> 1
                if bit == 1:
                    prod_left *= relations[idx]
                    prod_right *= roots[idx]
                idx += 1

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
        overall_start = time.time()
        self.logger.info("========== Quadratic Sieve V4 Start ==========")
        self.logger.info("Factoring N = %d", N)

        step_start = time.time()
        B = self.decide_bound(N, self.B)
        step_end = time.time()
        self.logger.info("Step 1 (Decide Bound) took %.3f seconds", step_end - step_start)

        step_start = time.time()
        prime_list = self.choose_multiplier(N, self.B)
        step_end = time.time()
        self.logger.info("Step 2 (Choose Multiplier) took %.3f seconds", step_end - step_start)
        kN = self.multiplier * N
        if kN.bit_length() < 140:
            LOWER_BOUND_SIQS = 3

        step_start = time.time()
        factor_base = self.build_factor_base(kN, B, prime_list)
        step_end = time.time()
        self.logger.info("Step 3 (Build Factor Base) took %.3f seconds", step_end - step_start)

        step_start = time.time()
        matrix, relations, roots = self.sieve(kN, B, factor_base, self.M)
        step_end = time.time()
        self.logger.info("Step 4 (Sieve Interval) took %.3f seconds", step_end - step_start)

        n = len(relations)
        step_start = time.time()
        null_space = self.solve_bits(matrix, n)
        step_end = time.time()
        self.logger.info("Step 5 (Solve Dependencies) took %.3f seconds", step_end - step_start)

        step_start = time.time()
        f1, f2 = self.extract_factors(N, relations, roots, null_space)
        step_end = time.time()
        self.logger.info("Step 6 (Extract Factors) took %.3f seconds", step_end - step_start)

        if f1 and f2:
            self.logger.info("Quadratic Sieve successful: %d * %d = %d", f1, f2, N)
        else:
            self.logger.warning("No non-trivial factors found with the current settings.")

        overall_end = time.time()
        self.logger.info("Total time for Quadratic Sieve: %.10f seconds", overall_end - overall_start)
        self.logger.info("========== Quadratic Sieve End ==========")

        return f1, f2

if __name__ == '__main__':

    ## 60 digit number
    #N = 373784758862055327503642974151754627650123768832847679663987
    #qs = QuadraticSieve(B=111000, M=400000, T=10, prime_limit=45, eps=34, lp_multiplier=20000)

    ### 70 digit number
    #N = 3605578192695572467817617873284285677017674222302051846902171336604399
    #qs = QuadraticSieve(B=300000, M=350000, prime_limit=47, eps=40, T=10, lp_multiplier=256)

    # 80 digit number
    N = 4591381393475831156766592648455462734389 * 1678540564209846881735567157366106310351
    qs = QuadraticSieve(B=700_000, M=600_000, prime_limit=52, eps=45, T=10, lp_multiplier=256)

    factor1, factor2 = qs.factor(N)
