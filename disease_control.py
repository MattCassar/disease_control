import numpy as np 
from matplotlib import pyplot as plt 
from simulate import simulate_viral_population, simulate_random_graphs
import utils

def simulate_random_conditions(n=4, m=3):
    M = utils.generate_random_mutation_graph(n=n)
    I = np.eye(n)
    D = np.random.uniform(0, 1, (n, m))

    MAX_TREATMENT = 0.9
    MIN_TREATMENT = 0.3
    for i in range(m):
        D[:, i] = np.random.uniform(MIN_TREATMENT, MAX_TREATMENT)*D[:, i]/sum(D[:, i])

    x0 = [10**np.random.randint(-3, 4) for _ in range(n)]
    delta = np.random.uniform(-.5, 0)
    mu = np.random.uniform(10**-6, 10**-3)

    fname = "simulations/random_simulation_" + "_".join(str(x) for x in x0)
    utils.save_simulation_conditions(M, D, x0, delta, mu, m, n, fname=fname)
    generate_results(M, D, x0, delta, mu, m, n)


def generate_eig_plots(M, D, x0, delta, mu, n, title="", num_iter=1000):
    lambdas, x, y = simulate_random_graphs(M, D, x0, delta, mu, n, num_iter=num_iter)
    utils.create_eigval_scatter(x, y)
    utils.create_eigval_histogram(lambdas, title=title)


def reproduce_paper_results():
    # Initial Conditions
    t_step, num_days = 2, 200
    time = [t*t_step for t in range(int(np.ceil(num_days/t_step)))]

    m, n = 3, 4
    delta, mu = -0.24, 10**(-4)

    I = np.eye(n)
    M = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])

    D = np.array([
        [0.05, 0.10, 0.30],
        [0.25, 0.05, 0.30],
        [0.10, 0.30, 0.30],
        [0.30, 0.30, 0.15]
    ])

    x0 = [10**3, 10**(-1), 10**(-1), 10**(-3)]

    generate_results(M, D, x0, delta, mu, m, n)


def generate_results(M, D, x0, delta, mu, m, n, t_step=2, num_days=200):
    I = np.eye(n)
    A = delta*I + mu*M

    time = [t*t_step for t in range(int(np.ceil(num_days/t_step)))]

    # Simulate and plot
    x, u, lambda_pf = simulate_viral_population(x0=x0, time=time, A=A, D=D)
    z = {}
    handles = ["x"+str(i) for i in range(1, n + 1)]

    for x_i in x:
        z[x_i] = np.log10(x[x_i])

    xlabel, ylabel = "time [days]", "population"
    top, bottom = 3.1, -5
    ylim = (top, bottom)
    utils.create_figure(time, z, handles=handles, xlabel=xlabel, ylabel=ylabel, 
                        title="Viral Population vs Time", ylim=ylim,
                        num_items=n)

    handles = ["u"+str(i) for i in reversed(range(1, m + 1))]
    top, bottom = 1, 0
    ylim = (top, bottom)

    ylabel = "relative concentration"
    utils.create_figure(time, u, handles=handles, xlabel=xlabel, ylabel=ylabel, 
                        title="Treatment Concentration vs. Time", ylim=ylim, 
                        num_items=m)

    title = "Perron-Frobenius Eigenvalue Distribution for Random Graphs"
    generate_eig_plots(M, D, x0, delta, mu, n, title=title)