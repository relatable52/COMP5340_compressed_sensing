# %% [markdown]
# *Computer Assignments: Sparse Recovery via $l_0$ and $l_1$ minimization*
# 
# **Recovery Challenge:** We have a signal $\mathbf{x}$ of 100 samples ($N = 100$) where no more than 3 of these samples are nonzero ($S \leq 3$). The location and magnitude of these nonzero samples are unknown. We have applied two different sensing matrices $\mathbf{A}$ on $\mathbf{x}$ and obtained two set of measurements $\mathbf{y} = \mathbf{A}\mathbf{x}$. The data is downloaded in the same folder as this notebook.

# %%
# Import necessary libraries
from itertools import combinations
import time

from scipy.io import loadmat
import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

# %% [markdown]
# First we need to load the sensing matrices from the .mat file.

# %%
DATA_FILE = '../data/COMP5340HW1.mat'

# %%
def load_data(data_file):
    """ 
    Load the data from the .mat file and return it as a dictionary.
    """
    data = loadmat(data_file)

    Af = data['Af']
    Ar = data['Ar']
    yf = data['yf']
    yr = data['yr']
    
    return {
        'Af': Af,
        'Ar': Ar,
        'yf': yf,
        'yr': yr
    }

# data = load_data(DATA_FILE)
# print("Data loaded successfully. The shapes of the matrices are:")
# print(f" Af: {data['Af'].shape}, Ar: {data['Ar'].shape}, yf: {data['yf'].shape}, yr: {data['yr'].shape}")

# %% [markdown]
# ## Part 1
# Recover $\mathbf{x}$ using $l_0$ minimization via exhaustive search.

# %%
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

# %%
# x_l0 = solve_l0_exhaustive(data['Af'], data['yf'])
# solution_found = None
# for sol in x_l0:
#     if check_solution(data['Ar'], sol, data['yr'].ravel()):
#         print("Found a valid solution using l0 minimization: \n %s" % sol)
#         solution_found = sol
# if solution_found is None:
#     print("No valid solution found using l0 minimization.")

# %% [markdown]
# ## Part 2
# Recover $\mathbf{x}$ using $l_1$ minimization via linear programming.

# %%
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
        print("Linear programming failed to find a solution: %s" % res.message)
        return None

# %%
# x_l1 = solve_l1_linear_programming(data['Af'], data['yf'])
# if x_l1 is not None and check_solution(data['Ar'], x_l1, data['yr'].ravel()):
#     print("Found a valid solution using l1 minimization: \n", x_l1)
# else:
#     print("No valid solution found using l1 minimization.") 

# %% [markdown]
# We can see that there are very small difference between the vector recovered by $l_0$ minimization and $l_1$ minimization. So small they are essentially the same. 
# 
# Now we rerun and time the 2 strategies.

# %%
def timing_wrapper(func):
    """
    A simple timing wrapper to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Execution time for %s: %f seconds" % (func.__name__, end_time - start_time))
        return result
    return wrapper

# %%
# print("\nTiming the l0 minimization solver:")
# x_l0_timed = timing_wrapper(solve_l0_exhaustive)(data['Af'], data['yf'])

# print("\nTiming the l1 minimization solver:")
# x_l1_timed = timing_wrapper(solve_l1_linear_programming)(data['Af'], data['yf'])

# %% [markdown]
# We can see that the $l_1$ minimization via linear programming is much faster.
# 
# ## Part 3
# 
# We see that both methods arrive at similar results. 
# 
# Both matrices successfully recovered the signal for sparsity $S=3$.
# 
# Taking a closer look at the recovered vectors. For that we will use a stem plot.

# %%
# Plot the solutions for visualization
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.stem(np.linspace(0, len(x_l0[0]), len(x_l0[0])), x_l0[0])
# plt.title('l0 Minimization Solution')
# plt.subplot(1, 2, 2)
# plt.stem(np.linspace(0, len(x_l1), len(x_l1)), x_l1)
# plt.title('l1 Minimization Solution')
# plt.show()

# %% [markdown]
# Let's take a closer look at the 2 sensing matrices.

# %%
# print Af
# A_f = np.array2string(data['Af'], precision=3, suppress_small=True)
# print(f"Af shape: {data['Af'].shape}")
# print("Af:\n%s" % data['Af'])


# print Ar
# A_r = np.array2string(data['Ar'], precision=3, suppress_small=True)
# print(f"Ar shape: {data['Ar'].shape}")
# print("Ar:\n%s" % data['Ar'])

# %% [markdown]
# At first glance, there is not much difference between the matrices.
# 
# We can try plotting the two matrices.

# %%
# Plot the 2 matrices as heatmaps
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.title("Sensing Matrix Af")
# plt.imshow(data['Af'], aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.title("Sensing Matrix Ar")
# plt.imshow(data['Ar'], aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.tight_layout()

# %% [markdown]
# It looks like the rows of sensing matrix $\mathbf{A_f}$ are sine waves of different frequency and the values of items in sensing matrix $\mathbf{A_r}$ are random.

# %% [markdown]
# ## Part 4
# 
# Testing for different sparsity level.
# 
# Since $l_0$ minimization takes a long time to run. We will not test it as the computation time will increase factorially.

# %%
def generate_and_test(A, S_level, seed=42):
    N = A.shape[1]
    np.random.seed(seed)
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

# %%
# for S in range(3, 11):
#     error = generate_and_test(data['Af'], S)
#     print(f"Sparsity level: {S}, Recovery error: {error}")

# %% [markdown]
# To see how the recovery error is affected by the sparsity level, we can perform the experiment many times and calculate the average error for each sparsity level.

# %%
# errors = {}
# for S in range(3, 15):
#     for seed in range(50):
#         error = generate_and_test(data['Af'], S, seed)
#         errors[S] = errors.get(S, []) + [error]

# Plot boxplot of error vs sparsity level
# plt.figure(figsize=(10, 6))
# plt.boxplot([errors[S] for S in errors], positions=range(3, 15))
# plt.xlabel("Sparsity Level (S)")
# plt.ylabel("Recovery Error")
# plt.title("Recovery Error vs Sparsity Level")
# plt.show()

# %% [markdown]
# It can be observed that the higher the sparsity level, the higher the chances that we get a high recovery errors. It is also worth noted that the error increases significantly at sparsity level 8.


