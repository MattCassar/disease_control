import numpy as np 
import cvxpy as cp 

def find_optimal_dosage(A, D, x, m=3, n=4):
    z = np.log10(np.array(x))

    # treatment concentration vectors
    u = cp.Variable((m, 1)) 

    # Perron-Frobenius eigenvalue of ODE system
    lambda_pf = cp.Variable((1, 1)) 

    # Minimize the Perron-Frobenius eigenvalue to maximize rate of viral decay
    objective = cp.Minimize(lambda_pf)
    constraints = []

    # Treatment concentrations must be non-negative and lie in unit simplex
    constraints.append(u >= 0)
    constraints.append(sum(u) == 1)

    for k in range(n):
        constraints.append(A[k, :]@cp.exp(z - z[k]) + D[k, :]@u <= lambda_pf)

    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    return lambda_pf, u


def find_optimal_varying_dosage(A, D, x0, m=3, n=4):
    raise Exception("Not currently implemented")
    
    z = np.log10(np.array(x))

    # treatment concentration vectors
    u = cp.Variable((m, 1)) 

    objective