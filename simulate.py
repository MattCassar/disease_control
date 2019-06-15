import numpy as np
import optimize
from collections import defaultdict
import utils

def forward_euler_step(f, x, t_step=None):
    if t_step is None:
        t_step = 2

    return x + t_step * f(x)


def find_D_weighted(u, D, n):
    D_weighted = np.zeros((n, n))

    for i in range(u.value.shape[0]):
        D_weighted += u.value[i]*np.diag(D[:, i])

    return D_weighted


def simulate_viral_population(x0=None, time=None, A=None, D=None, 
                              vary_dosage=False, t_step=None):
    population, concentration = defaultdict(list), defaultdict(list)
    x = x0

    if not vary_dosage:
        lambda_pf, u = optimize.find_optimal_dosage(A, D, x)

    for t in time:
        if vary_dosage:
            lambda_pf, u = optimize.find_optimal_dosage(A, D, x)

        for i, x_i in enumerate(x):
            population[i].append(x_i)

        for i, u_i in enumerate(u.value):
            concentration[i].append(u_i)

        D_weighted = find_D_weighted(u, D, len(x))
        x_dot = lambda x: (A + D_weighted)@x
        x = forward_euler_step(x_dot, x, t_step=t_step)

    return population, concentration, lambda_pf.value


def simulate_random_graphs(M, D, x0, delta, mu, n, num_iter=1000):
    lambdas = []
    lambda_orig, lambda_transpose = [], []

    for _ in range(num_iter):
        M = utils.generate_random_mutation_graph()

        A = delta*np.eye(n) + mu*M
        lambda_pf, _ = optimize.find_optimal_dosage(A, D, x0)

        lambdas.append(lambda_pf.value[0, 0])
        lambda_orig.append(lambda_pf.value[0, 0])

        A = delta*np.eye(n) + mu*M.T
        lambda_pf, _ = optimize.find_optimal_dosage(A, D, x0)
        
        lambdas.append(lambda_pf.value[0, 0])
        lambda_transpose.append(lambda_pf.value[0, 0])

    return lambdas, lambda_orig, lambda_transpose