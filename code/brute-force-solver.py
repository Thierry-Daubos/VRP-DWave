#!/usr/bin/env python
"""This program implements the brute-force technique to solve the TSP
(Traveling Salesman Problem)

The algorithm is simple. We simply enumerate all possible combinations and check
which one will minimize the cost function.

The program takes three inputs:
1. A file containing the pairwise costs as a matrix. These can be found in data.zip
2. The output file where the costs of all combinations are written (for debugging etc.)
3. The output file to store the solution found by this method.

The program can be run like this:
$ python brute-force-solver.py problem.txt scores.txt solution.txt
"""

import numpy as np
import sys
import itertools
import time

__author__ = "Siddharth Jain"
__copyright__ = "Copyright 2021, Johnson & Johnson"
__credits__ = ["Siddharth Jain"]
__license__ = "Apache 2.0"
__version__ = "1.0.1"
__maintainer__ = "Siddharth Jain"
__email__ = "sjain68@its.jnj.com"
__status__ = "Production"

def ring(nodes):
    """ create a ring from input list that tells which node is connect to which. The input list should be thought of as a ring or cycle. """
    n = len(nodes)
    A = np.zeros((n,n))
    for i in range(n):
        j = (i + 1) % n
        u = nodes[i]
        v = nodes[j]
        A[u,v] = A[v,u] = 1 # connect nodes[i] to nodes[j]
    return A

def enumerate_all_rings(n):
    """ Enumerate all combinations (possible ways) by which n cities (nodes) can be traversed. The # of combinations is (n-1)!/2 """
    for p in itertools.permutations(range(n-1)):
        if p[0] <= p[-1]:
            # add the n-th element to the array
            nodes = list(p)
            nodes.append(n-1)
            yield ring(nodes)

in_file = sys.argv[1]
out_file1 = sys.argv[2]
out_file2 = sys.argv[3]

# the matrix of pairwise costs
# this need not be a symmetric matrix but the diagonal entries are ignored and assumed to be zero

M = np.loadtxt(in_file)
n, _ = M.shape
k = 0
best_matrix = np.zeros((n,n))
best_score = np.inf
second_best_score = np.inf
unique_solutions = []
t0 = time.perf_counter()
with open(out_file1, 'w') as f:
    for A in enumerate_all_rings(n):
        # multiply will do element-wise multiplication
        # sum will sum over all the elements
        score = np.sum(np.multiply(M, A))
        f.write("{0} {1}\n".format(k, score))
        k +=1
        if score == best_score:
            unique_solutions.append(A)
        elif score < best_score:
            second_best_score = best_score
            best_score = score
            best_matrix = A
            unique_solutions = [A]
        elif score < second_best_score:
            second_best_score = score
t1 = time.perf_counter()

with open(out_file2, 'w') as f:
    f.write("Best score: {0}\n".format(best_score))
    f.write("Number of distinct solutions: {0}\n".format(len(unique_solutions)))
    for solution in unique_solutions:
        f.write("{0}\n".format(solution))
    f.write("Second best score: {0}\n".format(second_best_score))
    f.write("Energy difference: {0}\n".format(best_score - second_best_score))
    f.write(f"Time: {t1-t0:0.4f} s\n")