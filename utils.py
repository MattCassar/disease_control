import numpy as np 
from matplotlib import pyplot as plt
import os

SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def generate_random_mutation_graph(n=4):
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = np.random.randint(0, 2) if i != j else 0

    return M


def generate_random_symmetric_mutation_graph(n=4):
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            M[i, j] = np.random.randint(0, 2)
            M[j, i] = M[i, j]

    return M


def create_figure(x, y, handles=None, xlabel="", ylabel="", title="", 
                  xlim=None, ylim=None, num_items=1):
    plt.figure()

    for i in range(num_items):
        plt.plot(x, y[i])

    if handles is not None:
        plt.legend(handles)

    if xlabel != "":
        plt.xlabel(xlabel)

    if ylabel != "":
        plt.ylabel(ylabel)

    if title != "":
        plt.title(title)

    if xlim is not None:
        left, right = xlim
        plt.xlim(left=left, right=right)

    if ylim is not None:
        top, bottom = ylim
        plt.ylim(top=top, bottom=bottom)


def create_eigval_histogram(pf_evals, title="", bins=30,
                            xlabel="Frequency",
                            ylabel="Perron-Frobenius Eigenvalue"):
    plt.figure()
    plt.hist(pf_evals, bins=bins)

    if title != "":
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def create_eigval_scatter(orig_evals, transpose_evals):
    plt.figure()
    plt.scatter(orig_evals, transpose_evals)

    title = '''Perron-Frobenius Eigenvalue of M vs. 
        M^T ({} Random Graphs'''.format(len(orig_evals))

    plt.xlabel("Perron-Frobenius Eigenvalue of M")
    plt.ylabel("Perron-Frobenius Eigenvalue of M^T")


def conditions_to_dict(M, D, x0, delta, mu, m, n):
    return {"M": M, "D": D, "x0": x0, "delta": delta, "mu": mu, "m": m, "n": n}


def save_simulation_conditions_txt(fname, **kwargs):
    write_arg = "w"

    if os.path.isfile(fname):
        write_arg = "w+"

    with open(fname, write_arg) as file:
        for condition in sorted(kwargs.keys()):
            file.write("{}: {}\n".format(condition, kwargs[condition]))
        else:
            file.write("\n")


def save_simulation_conditions_csv(fname, **kwargs):
    write_arg = "w"

    if os.path.isfile(fname):
        write_arg = "w+"

    with open(fname, write_arg) as file:
        for condition in sorted(kwargs.keys()):
            file.write("{},")
        else:
            file.write("\n")


def save_simulation_conditions(M, D, x0, delta, mu, m, n, fname=""):
    conditions = conditions_to_dict(M, D, x0, delta, mu, m, n)
    save_simulation_conditions_txt(fname + ".txt", **conditions)
    save_simulation_conditions_csv(fname + ".csv", **conditions)