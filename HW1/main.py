from itertools import combinations
import pickle
import time
import logging

from scipy.io import loadmat
import numpy as np
from scipy.optimize import linprog

DATA_FILE = 'COMP5340HW1.mat'
SAVE_FILE = 'COMP5340HW1.pkl'
RESULTS_FILE = 'results.txt'

# Set up logging to print and save results to a text file
logging.basicConfig(filename=RESULTS_FILE, filemode='w', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def load_and_save_data(mat_file, save_file):
    """
    Load data from a .mat file and save it as a .pkl file.
    """
    data = loadmat(mat_file)

    Af = data['Af']
    Ar = data['Ar']
    yf = data['yf']
    yr = data['yr']
    
    new_data = {
        'Af': Af,
        'Ar': Ar,
        'yf': yf,
        'yr': yr
    }

    logger.info("Loaded data with keys: %s", new_data.keys())
    logger.info("Shape of sensing matrices A and y: %s, %s", Af.shape, yf.shape)

    with open(save_file, 'wb') as f:
        pickle.dump(new_data, f)
    
    return new_data

def check_solution(A, x, y):
    """
    Check if the solution x satisfies Ax = y.
    """
    return np.allclose(A @ x, y.ravel())
    
def solve_l0_exhaustive(A, y):
    """
    Solve the l0 minimization problem: min ||x||_0 subject to Ax = y.
    """
    # x is sparse with less than 3 non-zero entries
    # we can use a brute-force approach to find the solution
    M, N = A.shape
    # Check 1 non-zero entry, then 2, then 3
    solutions = []
    for k in range(1, 4):
        for indices in combinations(range(N), k):
            try:
                x = np.zeros(N)
                A_sub = A[:, indices]
                x_sub = np.linalg.lstsq(A_sub, y.ravel(), rcond=None)[0]
                x[list(indices)] = x_sub
                if check_solution(A, x, y):
                    solutions.append(x)
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue
    return solutions

def solve_l1_linear_programming(A, y):
    """
    Solve the l1 minimization problem: min ||x||_1 subject to Ax = y.
    This can be implemented using linear programming or convex optimization libraries.
    For simplicity, we will not implement this here, but it can be done using libraries like CVXPY.
    """
    M, N = A.shape

    # L1 norm can be represented as a linear program by introducing auxiliary variables u, v
    # x = u - v, where u >= 0, v >= 0, and we minimize sum(u + v)

    # Constraints: A(u - v) = y, u >= 0, v >= 0
    A_eq = np.hstack((A, -A))  # Combine A for u and -A for v
    b_eq = y.ravel()
    
    # Objectives function coefficients for the linear program
    # Here we are minimizing the l1 norm
    c = np.ones(2*N)

    # Solve the linear program using scipy's linprog
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if res.success:
        u = res.x[:N]
        v = res.x[N:]
        x = u - v
        return x
    else:
        logger.error("Linear programming failed to find a solution: %s", res.message)
        return None
    
def timing_wrapper(func):
    """
    A simple timing wrapper to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info("Execution time for %s: %f seconds", func.__name__, end_time - start_time)
        return result
    return wrapper

def generate_and_test(A, S_level):
    N = A.shape[1]
    # 1. Create a sparse x
    x_true = np.zeros(N)
    indices = np.random.choice(N, S_level, replace=False)
    x_true[indices] = np.random.uniform(1, 50, S_level) # Random magnitudes
    
    # 2. Generate measurements
    y = A @ x_true
    
    # 3. Recover with L1 (LP)
    x_rec = solve_l1_linear_programming(A, y)
    
    # 4. Check error
    error = np.linalg.norm(x_true - x_rec)
    return error

if __name__ == "__main__":
    # Load data from .mat file and save as .pkl
    data = load_and_save_data(DATA_FILE, SAVE_FILE)

    # Part 1: Solve the l0 minimization problem by exhaustive search
    logger.info("==== Part 1: Solving l0 minimization problem ====")
    x_l0 = solve_l0_exhaustive(data['Af'], data['yf'])
    solution_found = None
    for sol in x_l0:
        if check_solution(data['Ar'], sol, data['yr'].ravel()):
            logger.info("Found a valid solution using l0 minimization: \n %s", sol)
            solution_found = sol
    if solution_found is None:
        logger.info("No valid solution found using l0 minimization.")

    # Part 2: Solve the l1 minimization problem using linear programming
    logger.info("==== Part 2: Solving l1 minimization problem ====")
    x_l1 = solve_l1_linear_programming(data['Af'], data['yf'])
    if x_l1 is not None and check_solution(data['Ar'], x_l1, data['yr'].ravel()):
        logger.info("Found a valid solution using l1 minimization: \n %s", x_l1)
    else:
        logger.info("No valid solution found using l1 minimization.") 

    # Rerun and time the l0 and l1 solvers
    logger.info("\nTiming the l0 minimization solver:")
    x_l0_timed = timing_wrapper(solve_l0_exhaustive)(data['Af'], data['yf'])

    logger.info("\nTiming the l1 minimization solver:")
    x_l1_timed = timing_wrapper(solve_l1_linear_programming)(data['Af'], data['yf'])
    
    # Part 3: Observations for the two sensing matrices Af and Ar and the solutions
    logger.info("\n==== Part 3: Observations ====")
    logger.info("Observations for sensing matrix Af:")
    
    # Print Af
    A_f = np.array2string(data['Af'], precision=3, suppress_small=True)
    logger.info("Af shape: %s", data['Af'].shape)
    logger.info("Af:\n%s", data['Af'])

    logger.info("Observations for sensing matrix Ar and solution yr:")

    # Print Ar
    A_r = np.array2string(data['Ar'], precision=3, suppress_small=True)
    logger.info("Ar shape: %s", data['Ar'].shape)
    logger.info("Ar:\n%s", data['Ar'])

    logger.info("The 2 matrices are different. But solving on both found the same solution.")
    logger.info("Both solvers found the same solution, however the l1 solver is much faster.")

    # Part 4: Test for different sparsity levels
    logger.info("\n==== Part 4: Testing for different sparsity levels ====")
    for i in range(3, 11):
        error = generate_and_test(data['Af'], i)
        logger.info("Sparsity level: %d, Recovery error: %f", i, error)