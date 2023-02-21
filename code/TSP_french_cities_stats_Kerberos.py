# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:37:52 2022

@author: thierry.daubos
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import random
import copy
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import networkx.algorithms.approximation as nx_app

from IPython import get_ipython
from datetime import datetime
from mpl_toolkits.basemap import Basemap as Basemap
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

import minorminer
import dwave.inspector
import dwave_networkx as dnx

from dwave import embedding
from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding.chain_strength import scaled
from dwave.system import FixedEmbeddingComposite
from dwave.system.composites import LazyFixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system import VirtualGraphComposite
from dwave.system import DWaveCliqueSampler
from hybrid import KerberosSampler

import dimod
import neal
from minorminer import find_embedding
from minorminer import busclique

__author__     = "Thierry Daubos"
__copyright__  = "Copyright 2022, Scalian DS"
__credits__    = ["Thierry Daubos"]
__license__    = "Apache 2.0"
__version__    = "1.0.0"
__maintainer__ = "Thierry Daubos"
__email__      = "thierry.daubos@scalian.com"
__status__     = "developpment"

#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

def traveling_salesperson_qubo(G, lagrange = None, weight = 'weight'):
    """Return the QUBO with ground states corresponding to a minimum TSP route.

    If :math:`|G|` is the number of nodes in the graph, the resulting qubo will have:

    * :math:`|G|^2` variables/nodes
    * :math:`2 |G|^2 (|G| - 1)` interactions/edges

    Parameters
    ----------
    G : NetworkX graph
        A complete graph in which each edge has a attribute giving its weight.

    lagrange : number, optional (default None)
        Lagrange parameter to weight constraints (no edges within set)
        versus objective (largest set possible).

    weight : optional (default 'weight')
        The name of the edge attribute containing the weight.

    Returns
    -------
    QUBO : dict
       The QUBO with ground states corresponding to a minimum travelling
       salesperson route. The QUBO variables are labelled `(c, t)` where `c`
       is a node in `G` and `t` is the step index. For instance, if `('a', 0)`
       is 1 in the ground state, that means the node 'a' is visted first.

    """
    N = G.number_of_nodes()

    if lagrange is None:
        # If no lagrange parameter provided, set to 'average' tour length.
        # Usually a good estimate for a lagrange parameter is between 75-150%
        # of the objective function value, so we come up with an estimate for 
        # tour length and use that.
        if G.number_of_edges()>0:
            lagrange = G.size(weight=weight)*G.number_of_nodes()/G.number_of_edges()
        else:
            lagrange = 2

    # some input checking
    if N in (1, 2) or len(G.edges) != N*(N-1)//2:
        msg = "graph must be a complete graph with at least 3 nodes or empty"
        raise ValueError(msg)

    # Creating the QUBO
    Q           = defaultdict(float)

    # Constraint that each row has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += 2.0 * lagrange
    # linear coefficients
    # 5 * 25 = 75 lines -> 75
    
    # Constraint that each col has exactly one 1
    for pos in range(N):
        for node_1 in G:
            Q[((node_1, pos), (node_1, pos))] -= lagrange
            for node_2 in set(G)-{node_1}:
                # QUBO coefficient is 2*lagrange, but we are placing this value 
                # above *and* below the diagonal, so we put half in each position.
                Q[((node_1, pos), (node_2, pos))] += lagrange
    # quadratic coefficients corresponding to "each column has exactly one 1" constraint
    # 5 * (5*4) = 100 lines -> 175
    
    # Objective that minimizes distance
    for u, v in itertools.combinations(G.nodes, 2):
        for pos in range(N):
            nextpos = (pos + 1) % N

            # going from u -> v
            Q[((u, pos), (v, nextpos))] += G[u][v][weight]

            # going from v -> u
            Q[((v, pos), (u, nextpos))] += G[u][v][weight]
    # quadratic coefficients corresponding to minimum distance objective
    # 5 * (5*4) = 100 lines -> 275

    return Q

def is_hamiltonian_path(G, route):
    """Determines whether the given list forms a valid TSP route.

    A travelling salesperson route must visit each city exactly once.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to check the route.
    route : list
        List of nodes in the order that they are visited.
    Returns
    -------
    is_valid : bool
        True if route forms a valid travelling salesperson route.
    """
    return (set(route) == set(G))

def index(i, j, n):
    if i == j:
        raise ValueError
    elif i > j:
        return index(j, i, n)
    else:
        return int(i*n - (i*(i+1)/2) + j - (i+1))

def build_objective_matrix(M):
    n, _ = M.shape
    # m is the total number of binary variables that we have to solve for
    # basically given any two nodes, we need to decide if there is an edge connecting them (a binary variable)
    m = int(n*(n-1)/2)
    Q = np.zeros((m,m))
    k = 0
    for i in range(0, n):
        for j in range(i+1, n):
            # M[i,j] + M[j,i] is the cost to travel from i to j (or vice-versa)
            Q[k, k] = (M[i,j] + M[j,i])
            k += 1
    # Q is the diagonal matrix of biases
    return Q

def build_constraint_matrix(n, bias_value = -2.0, off_diag_bias = 4.0):
    """
     The constraints matrix encodes the constraint that each city (node) is connected to exactly two other cities in the output cycle        
    """
    m          = int(n*(n-1)/2)
    C          = np.zeros((m,m))
    for i in range(0, n):
        
        # Diagonal terms of C are equal to 2 * bias_value
        for j in range(0, n):
            if i == j:
                continue
            k = index(i, j, n)
            C[k,k] += bias_value
            
        # Off diagonal terms (the pattern for these elements is strange)
        for a in range(0, n):
            for b in range(0, n):
                if a == b or a == i or b == i:
                    continue
                ia = index(i,a,n)
                ib = index(i,b,n)
                # print("i : ", i, " a : ", a, " b : ", b, " index(", i,",",a,") :", ia, " index(",i,",",b,") :", ib)
                C[ia,ib] += off_diag_bias
    return C

def build_decode(M, city_names):
    '''
    Build a dictionnary of correspondances between edges index and connected cities
    Used for decoding the QPU solution returned
    '''
    n, _   = M.shape
    k      = 0
    decode = dict()
    
    for i in range(0, n):
        for j in range(i+1, n):
            decode[k] = [city_names[i], city_names[j]]
            k += 1
    return decode

def is_valid_solution(X):
    # Checks that each node is connected to only 2 other nodes
    rows, cols = X.shape
    for i in range(rows):
        count = 0
        for j in range(cols):
            if X[i,j] == 1:
                count += 1
        if not count == 2:
            return False

    # Checks that each node is visited only once and all nodes are visited    
    quantum_route   = decode_route(X)
    if not(len(set(quantum_route[:-1])) == len(quantum_route[:-1]) == Nb_cities):
        return False
    return True
    
def build_solution(sample):
    # This will use the global M variable
    n, _ = M.shape 
    m    = len(sample)
    assert m == int(n*(n-1)/2)
    X = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            X[i,j] = X[j,i] = sample[k]
            k += 1
    return X

def compute_score(M, X):
    return np.sum(np.multiply(M, X))

def decode_quantum_route(sample):
    ''' decode quantum sample result into route '''
    edge_list = list()
    for i, city in enumerate(sample):
        if sample[i] :
            edge_list.append(Decode[i])
            
    quantum_route = list()
    quantum_route.append(city_names.index(edge_list[0][0]))
    quantum_route.append(city_names.index(edge_list[0][1]))
    firt_city = edge_list[0][0]
    next_city = edge_list[0][1]
    edge_list.pop(0)
    
    while len(edge_list) > 1:
        for i, edge in enumerate(edge_list):
            if next_city in edge:
                if next_city != edge_list[i][0]:
                    quantum_route.append(city_names.index(edge_list[i][0]))
                    next_city = edge_list[i][0]
                else:
                    quantum_route.append(city_names.index(edge_list[i][1]))
                    next_city = edge_list[i][1]
                edge_list.pop(i)
    quantum_route.append(city_names.index(firt_city))
    return quantum_route

def ring(nodes):
    ''' Creates a ring from input list that tells which node is connect to which. The input list should be thought of as a ring or cycle. '''
    n = len(nodes)
    A = np.zeros((n,n))
    for i in range(n):
        j = (i + 1) % n
        u = nodes[i]
        v = nodes[j]
        # Add connection from nodes[i] to nodes[j]
        A[u,v] = A[v,u] = 1
    return A

def enumerate_all_rings(n):
    ''' Enumerate all combinations that traverse the n cities (nodes). N° of combinations is (n-1)!/2 '''
    for p in itertools.permutations(range(n-1)):
        if p[0] <= p[-1]:
            # add the n-th element to the array
            nodes = list(p)
            nodes.append(n-1)
            yield ring(nodes)

def cycle2solution(cycle):
    n = len(cycle)-1
    sol = list()
    for i in range(n):
        vector = np.zeros(n)
        vector[cycle[i]]   = 1.
        vector[cycle[i+1]] = 1.
        sol.append(vector)
    sol = np.array(sol)
    
    return sol
            
def decode_route(solution):
    ''' decode Brute Force solution into route '''
    
    edge_list = list()
    for i in range(n):
        for j in range(i+1, n):
            if solution[i,j] == 1:
                edge_list.append([i,j])
    
    BF_route = list()
    found = False
    for i, e in enumerate(edge_list):
        if e[0] == 0 and not found:
         BF_route.append(e[0])
         BF_route.append(e[1])
         first_city = e[0]
         next_city  = e[1]
         found = True
         edge_list.pop(i)
    
    fail_safe = 0
    while len(edge_list) > 1 and fail_safe < 100:
        found = False
        for i, e in enumerate(edge_list):
            if e[0] == next_city and not found:
                 BF_route.append(e[1])
                 next_city = e[1]
                 found = True
                 edge_list.pop(i)
                 
            if e[1] == next_city and not found:
                 BF_route.append(e[0])
                 next_city = e[0]
                 found = True
                 edge_list.pop(i)
        fail_safe += 1
    
    BF_route.append(first_city)
    return BF_route

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

global COMPUTE_QUANTUM
COMPUTE_QUANTUM = True

'''
# Read date file of cities
'''
dir_name = "D:\Documents\Scalian\Quantum_Computing_2022\Datasets"
base_filename = "fr"
filename_suffix = "csv"

filename = os.path.join(dir_name, base_filename + '.' + filename_suffix)
df = pd.read_csv(filename)
city_names = list(df.columns.map(str))

'''
# Get locations Lon/Lat coordinates
'''

geolocator = Nominatim(user_agent="MyApp")
lats = list()
lons = list()
for name in city_names:
    location = geolocator.geocode(name)
    lats.append(location.latitude)
    lons.append(location.longitude)
    
offset = 1.0
min_lon = min(lons)- offset
max_lon = max(lons)+ offset
min_lat = min(lats)- offset
max_lat = max(lats)+ offset

'''
# Define problem's meta parameters
'''
global Nb_cities
Nb_cities      = 11

# Seed value used for ramdomly selecting the cities
Seed_city      = 0

# Scaling factor for the pairwaise cost matrix between nodes
scaling_factor = 1.0

# Bias value used to build the constraint matrix
# bias_value     = -2.0
# off_diag_bias  =  1.0
bias_value     = -1.0
off_diag_bias  =  2.0

# Number of measurements repetition (up to 10000)
num_samples    = 1000

# Sets annealing duration per sample in microseconds (up to 2000)
# annealing_time = 1000
annealing_time = 100

# Number of standard deviations used to compute chain strength
N_sigma        = 3.0

# Relative chain strength
# chain_strength = RCS * max_chain_strength # 'conservative' value 
# or
# chain_strength = RCS * int(mean_chain_strength + (N_sigma * std_chain_strength)) # 'tighter' value
# The (relative) chain strength to use in the embedding. 
# By default a chain strength of `1.5 sqrt(N)` where `N` is the size of the largest clique, as returned by attribute `.largest_clique_size`
RCS            = 1.00

# Lagrange multiplier for taking into account the constraint matrix

# Relative Lagrange Multiplier
RLM            = 1.00

# Reinitialize embedding
fixed_embedding = None

random.seed(Seed_city)
cities_keep    = random.sample(city_names, Nb_cities)
# cities_keep    = ['Paris', 'Lille', 'Aix-en-Provence', 'Nimes']
cities_removed = copy.deepcopy(city_names)

for city in cities_keep:
    cities_removed.remove(city)

cities_indices = list()
for city in cities_removed:
    cities_indices.append(city_names.index(city))

df.drop(cities_removed, axis=1, inplace=True)
df.drop(cities_indices, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

city_names = list(df.columns.map(str))
n_nodes    = len(df.columns)
city_index = list(range(0, n_nodes))
df.columns = list(range(0, n_nodes))

print("city names ;", city_names)

df = df.fillna(0)
df = df.astype(int)

''' The objective should be that elements of the qubo matrix are all below the chain strength value '''

''' Define model's meta-parameters '''

df_km          = copy.deepcopy(df)
scaler         = MinMaxScaler(feature_range=(0., scaling_factor))
df             = scaler.fit_transform(df)

'''
*** Define objective matrix of travel costs ***
# M is the matrix of pairwise costs (i.e. cost to travel from node i to node j)
# M need not be a symmetric matrix but the diagonal entries are ignored and assumed to be zero
'''
M                   = df
Q_qubo              = build_objective_matrix(M)
Decode              = build_decode(M, city_names)

''' 
*** Encode connectivity constraint in the final cycle ***
# We need to add the constraint that each city is connected to exactly 2 other cities
# We do this using the method of lagrange multipliers where the constraint is absorbed into the objective function
# (this is the hardest part of the problem)
'''
n, _       = M.shape
C          = build_constraint_matrix(n, bias_value, off_diag_bias)

'''
*** Define the Hamiltonian of the problem ***

H = HA + HB

HA : Hamiltonian component corresponding to the (directed or undirected) Hamiltonian cycle problem
HA = A . (term_1 + term_2 + term-3)
term_1 : constraint that every vertex can only appear once in a cycle      (i.e. all nodes must be visited at most once)
term_2 : constraint that "there must be a jth node in the cyle for each j" (i.e. all nodes need to be visited at least once)
term_3 : if Xu,j and Xv,j+1 are both 1 then there should be an energie penalty if (u,v) is not in the edgeset of the graph

HB : Hamiltonian component corresponding to the Weighted graph problem
HB = B . term_4

B sould be small enough i.e. 0 < B max(Wuv) < A
so that no solution can favour to violate the constraints of HA

* In this implementation * :
    
M                  <-> Wuv : Mij cost of travelling form i to j
Q_qubo             <-> HA  : Objective matrix Q = M.X (X mapping Xij to Bk)
C                  <-> HB  : Constraints matrix mapped to Bk
lagrange_mutiplier <-> (B/A)
qubo               <-> H
'''

''' Define NetworkX graph '''
G_embedding      = nx.complete_graph(n_nodes)
G_embedding.name = "Graph of " + str(n_nodes) + " french cities"

# Calculating the distances between the nodes as edge's weight.
for i in city_index:
    for j in range(i+1, n_nodes):
        G_embedding.add_edge(i, j, weight=df[i][j])

print("G_embedding ", G_embedding.number_of_nodes(), "Nodes: \n", list(G_embedding.nodes))
print("G_embedding ", G_embedding.number_of_edges(), "Edges: \n", list(G_embedding.edges))
for node in G_embedding.nodes:
    print(node,"->", G_embedding.nodes[node])

for edge in G_embedding.edges:
    print(edge,"->", G_embedding.edges[edge])

# RLM = 1.0
# base_lagrange       = G_embedding.size(weight = 'weight') * G_embedding.number_of_nodes() / G_embedding.number_of_edges()
# temp_lagrange_multiplier = RLM * base_lagrange

temp_lagrange_multiplier = 3.038346244970316
qubo = Q_qubo + temp_lagrange_multiplier * C

# print("Cost matrix M        : \n", M, "\n")
# print("Objective matrix Q   : \n", Q_qubo, "\n")
# print("Lagrange multiplier  : \n", temp_lagrange_multiplier, "\n")
# print("Constraints matrix C : \n", C, "\n")
# print("QUBO matrix qubo     : \n", qubo, "\n")

''' Load problem's solutions from brute-force algorithm '''

path                 = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/"
name_BF_all_pkl      = "solution_BF_all-cities_"      + str(Nb_cities) + "_seed_" + str(Seed_city)
ext                  = ".pkl"
in_file_BF_all_pkl   = path + name_BF_all_pkl + ext
spectrum_brute_force = pd.read_pickle(in_file_BF_all_pkl)
spectrum_brute_force.reset_index(drop=True, inplace=True)

spec_min = spectrum_brute_force["score BF"].min()
spec_max = spectrum_brute_force["score BF"].max()
normalized_spectrum_brute_force = pd.DataFrame((spectrum_brute_force["score BF"] - spec_min) / (spec_max - spec_min))

spectrum_res = 0.01
round_digit  = len(str(int(1./spectrum_res)))-1
spectrum_min = trunc(normalized_spectrum_brute_force.min(), decs = round_digit)
spectrum_max = trunc(normalized_spectrum_brute_force.max(), decs = round_digit)
nb_bin       = int(np.ceil((spectrum_max - spectrum_min)) / spectrum_res)

# solution = unique_solutions[0]
solution        = (spectrum_brute_force["solution"].iloc[0])[0]
BF_route        = decode_route(solution) 
best_score_BF   = (spectrum_brute_force["score BF"].iloc[0])

'''
*** Run the problem on the QPU recording execution times
'''
have_solution = False

path             = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/"
name_SA_all      = "solution_SA_all-cities_"       + str(Nb_cities) + "_seed_" + str(Seed_city)
name_SA_spec     = "solution_SA_spectrum_cities_"  + str(Nb_cities) + "_seed_" + str(Seed_city)
name_SA_map      = "solution_SA_map_cities_"       + str(Nb_cities) + "_seed_" + str(Seed_city)

ext1             = ".txt"
ext2             = ".png"

out_file_SA_spec = path + name_SA_spec + ext2
out_file_SA_map  = path + name_SA_map  + ext2

if COMPUTE_QUANTUM:
    max_chain_strength  = np.ceil(np.max(np.abs(G_embedding)))
    mean_chain_strength = np.ceil(np.mean(np.abs(G_embedding)))
    std_chain_strength  = np.ceil(np.std(np.abs(G_embedding)))
    
    # Setting chain strength value 
    chain_strength = RCS * int(mean_chain_strength + (N_sigma * std_chain_strength))  

    print("Nuber of samples          : ", num_samples)
    print("Annealing time            : ", annealing_time)
    print("Relative chain strength   : ", RCS)
    print("Targeted chain strength   : ", chain_strength)
    
    base_lagrange       = G_embedding.size(weight = 'weight') * G_embedding.number_of_nodes() / G_embedding.number_of_edges()
    lagrange_multiplier = RLM * base_lagrange
    print("Total weight sum of edges      : ", G_embedding.size(weight = 'weight'))
    print("Nodes : Number of nodes        : ", G_embedding.number_of_nodes())
    print("Edges : Number of edges        : ", G_embedding.number_of_edges())
    print("Base Lagrange multiplier       : ", base_lagrange)
    print("Relative Lagrange Multiplier   : ", RLM)
    print("Lagrange multiplier            : ", lagrange_multiplier)
    # lagrange_multiplier = None
    
    Q = traveling_salesperson_qubo(G_embedding, lagrange = lagrange_multiplier)

    ''' Chose a DWave QPU '''

    qpu       = DWaveSampler(solver={'topology__type': 'pegasus'})
    print("QPU topology                              : ", qpu.properties['topology']['type'])
    print("QPU chip_id                               : ", qpu.properties['chip_id'])
    print("QPU qpu_delay_time_per_sample             : ", qpu.properties['problem_timing_data']['qpu_delay_time_per_sample'])
    print("QPU annealing_time_range                  : ", qpu.properties['annealing_time_range'])
    print("QPU num_reads_range                       : ", qpu.properties['num_reads_range'])
    print("QPU decorrelation_max_nominal_anneal_time : ", qpu.properties['problem_timing_data']['decorrelation_max_nominal_anneal_time'])
    print("QPU decorrelation_time_range              : ", qpu.properties['problem_timing_data']['decorrelation_time_range'])
    print("QPU problem_run_duration_range            : ", qpu.properties['problem_run_duration_range'])
    print("QPU h_gain_schedule_range                 : ", qpu.properties['h_gain_schedule_range'])
    print("QPU extended_j_range                      : ", qpu.properties['extended_j_range'])
    print("QPU per_qubit_coupling_range              : ", qpu.properties['per_qubit_coupling_range'])

    ''' Find embedding using Minorminer '''
    
    # 7 cities - fixed_embedding = {(0, 0): (3466, 393, 3467, 302, 482, 3465), (0, 1): (3555, 155, 3556, 153, 154), (0, 2): (288, 289, 290, 287, 3496, 3495), (0, 3): (304, 305, 3750, 3751, 303), (0, 4): (423, 422, 3421, 424, 3721), (0, 5): (453, 3647, 454, 452, 3497), (0, 6): (3527, 3528, 498, 3526), (1, 0): (3662, 3661, 483, 3663, 364), (1, 1): (214, 213, 215), (1, 2): (230, 3886, 3825, 229), (1, 3): (319, 3720, 320, 318), (1, 4): (3872, 3871, 530, 545, 544), (1, 5): (3735, 664, 3737, 663, 3736), (1, 6): (228, 3512, 543, 3513, 3511), (2, 0): (3451, 3450, 363, 3452, 602), (2, 1): (168, 167, 3645, 169, 170, 3646), (2, 2): (349, 348, 347, 350, 3706, 3705), (2, 3): (274, 275, 273, 272, 3390, 197), (2, 4): (439, 438, 3826, 437), (2, 5): (528, 3632, 527, 3631, 529), (2, 6): (3407, 647, 648, 3406, 3405), (3, 0): (678, 3540, 3542, 3541, 679), (3, 1): (245, 244, 243, 242, 3975, 3976), (3, 2): (3946, 3945, 500, 3947, 680), (3, 3): (649, 3917, 3916, 3915, 3813), (3, 4): (499, 3857, 3856, 3858), (3, 5): (3677, 394, 3678, 708, 395), (3, 6): (618, 3708, 3977, 617, 619, 620), (4, 0): (3600, 3601, 3602, 3603, 93, 94), (4, 1): (257, 258, 259, 260, 3481, 3482), (4, 2): (3961, 515, 3962, 95, 3960), (4, 3): (365, 3811, 3812, 3810), (4, 4): (3842, 3841, 560, 3843, 3840), (4, 5): (3722, 514, 513, 512, 3707), (4, 6): (588, 589, 590, 3498, 3483), (5, 0): (3571, 3570, 3572, 378), (5, 1): (184, 185, 183, 3795, 3660), (5, 2): (199, 198, 3630, 200), (5, 3): (3676, 379, 380, 3675), (5, 4): (3781, 334, 3780, 333, 335, 3782), (5, 5): (559, 3692, 3693, 3691, 558, 3690), (5, 6): (3616, 3617, 3618, 3615, 108), (6, 0): (603, 3587, 3586, 3585, 604), (6, 1): (409, 408, 3766, 3765, 410), (6, 2): (485, 3888, 3887, 3931, 3930), (6, 3): (3901, 3900, 634, 3902), (6, 4): (3797, 3798, 3796, 484), (6, 5): (469, 467, 3827, 468), (6, 6): (574, 575, 573, 572, 3767)}
    # 8 cities - fixed_embedding = {(0, 0): (4393, 4391, 2062, 4392, 2242, 2063, 4394), (0, 1): (4481, 4482, 4483, 4480, 2033, 4484, 2034), (0, 2): (2589, 2588, 4619, 2590, 2591, 4993, 2587, 4979), (0, 3): (2530, 2529, 2531, 2528, 4948, 2527, 2396), (0, 4): (2455, 4918, 4963, 4962, 2454, 2453, 2452), (0, 5): (2245, 4977, 4978, 2244, 2243, 4916), (0, 6): (2064, 2065, 4990, 4991, 4705, 4992, 4781), (0, 7): (4616, 4617, 2049, 4618, 2048, 2047, 2050), (1, 0): (4437, 4438, 4439, 4436, 4435), (1, 1): (4467, 2378, 4468, 4466, 4465, 2377), (1, 2): (2649, 2648, 4648, 2650, 2647), (1, 3): (2604, 2603, 2602, 4693, 2605), (1, 4): (2350, 4737, 2349, 4738, 2348, 2351, 4739), (1, 5): (4933, 4932, 2290, 2289, 2288, 4931, 4934), (1, 6): (4810, 4811, 4812, 4813, 1990, 1989, 1988), (1, 7): (4496, 4495, 4497, 1958, 4540, 4498, 2004), (2, 0): (4423, 4424, 4422, 4421, 2693, 2694, 2695, 2318), (2, 1): (2498, 4557, 4558, 2497, 2499, 2500), (2, 2): (2633, 2634, 2632, 2635, 4633), (2, 3): (2560, 2559, 2561, 4769, 5023), (2, 4): (4783, 4782, 2470, 2515, 4784, 2619), (2, 5): (4857, 2320, 4947, 4946, 4858, 4859, 2260), (2, 6): (4841, 4840, 2005, 4842, 4843, 2020, 4844), (2, 7): (4752, 2259, 2258, 2257, 4751, 4753, 4754), (3, 0): (4333, 4332, 2437, 2438, 4331, 2439, 2077), (3, 1): (4527, 2393, 4528, 4526, 4529, 2394), (3, 2): (2663, 4904, 4453, 2664, 2665, 2662, 4334), (3, 3): (2484, 4708, 4709, 2483, 2485), (3, 4): (4602, 4603, 4604, 2334, 2335, 4601, 4600), (3, 5): (4826, 4827, 4825, 2155, 4828, 2395, 1975), (3, 6): (4901, 2080, 2079, 2078, 4902, 4903, 4660), (3, 7): (4450, 1973, 1974, 2018, 4451, 4452, 4765), (4, 0): (4377, 4376, 4378, 4375, 2317, 4379, 2558), (4, 1): (4303, 2617, 2618, 4302, 4301, 4300), (4, 2): (2572, 2573, 2574, 4347, 4348, 4346, 4198, 2575), (4, 3): (2545, 2544, 2546, 2543, 5053, 5052, 5051, 2542), (4, 4): (2740, 4889, 4888, 4887, 4886, 2739, 4559, 2738), (4, 5): (2274, 2275, 2276, 2273, 5006, 5005, 2272), (4, 6): (2109, 2110, 2111, 2108, 2107), (4, 7): (2124, 2123, 2122, 2125, 2126), (5, 0): (4587, 4586, 2333, 2332), (5, 1): (4512, 4513, 4511, 4514, 2363, 2362), (5, 2): (4543, 2513, 4544, 4541, 2754, 4542), (5, 3): (2365, 4722, 4723, 2364, 4724), (5, 4): (4797, 4798, 2200, 2199, 4799, 2198), (5, 5): (2305, 4855, 4856, 2304, 2303, 2306), (5, 6): (2169, 2170, 2171, 2168, 4721), (5, 7): (2138, 4571, 4572, 2137, 2019, 4555), (6, 0): (4317, 4316, 2422, 2423, 2424, 4315), (6, 1): (2408, 4647, 2407, 4588, 4589, 4646), (6, 2): (2379, 4707, 4706, 4573, 4574, 2468), (6, 3): (4663, 4664, 4662, 2469, 4661), (6, 4): (4768, 4767, 2514, 2319, 2380, 2381), (6, 5): (2409, 4692, 4691, 2410, 4690), (6, 6): (4631, 4632, 4630, 2139, 2140), (6, 7): (2154, 4766, 2035, 2153, 2152), (7, 0): (2214, 2215, 2213, 2212, 4556), (7, 1): (2723, 4364, 4363, 4362, 4361, 2724, 4360), (7, 2): (4409, 4408, 4407, 2708, 2709, 2710, 4406), (7, 3): (4677, 4678, 4676, 4679, 4675), (7, 4): (4873, 4872, 4874, 2680, 2679, 2440, 4871, 4694), (7, 5): (2229, 2230, 2228, 4736, 2227, 4735), (7, 6): (2185, 2184, 4796, 2183, 2186), (7, 7): (2093, 2094, 2092, 2095, 4645)}
    # 9 cities - fixed_embedding = {(0, 0): (3654, 1984, 3655, 1983, 3653, 2014, 1593, 1592, 2015), (0, 1): (3774, 1879, 1878, 1877, 1876, 1880, 3474, 3880), (0, 2): (3372, 3371, 3370, 3369, 2041, 3368, 1637, 1638), (0, 3): (3476, 2132, 2131, 3477, 3478, 3479, 3475, 2133), (0, 4): (3884, 2626, 3881, 3883, 2629, 2628, 2627, 3882, 3373), (0, 5): (3747, 3746, 3745, 3748, 3749, 3744, 3743, 1639, 2134), (0, 6): (3642, 2268, 3596, 3643, 3644, 2269, 3595, 3594, 3593), (0, 7): (2104, 2103, 2105, 3806, 2102, 3775), (0, 8): (2059, 3610, 2058, 2057, 3609, 1758, 1757), (1, 0): (1924, 3729, 1923, 1922, 3728, 1925, 1926), (1, 1): (1834, 1833, 1832, 1831, 1835, 3789, 1836, 3969), (1, 2): (2450, 4063, 4062, 4061, 4060, 2449, 2448, 2447, 2446), (1, 3): (3416, 3417, 2462, 3418, 3415, 3414, 2222, 2221), (1, 4): (2375, 2374, 2373, 2372, 3972, 3898, 3971, 3970, 3899, 3868), (1, 5): (3822, 3821, 2404, 2403, 3823, 3820, 3819, 2402), (1, 6): (3793, 3792, 2464, 2463, 3791, 3790, 2465), (1, 7): (3552, 3551, 3550, 3549, 2283, 3548), (1, 8): (2044, 2043, 3835, 3834, 2045, 2042), (2, 0): (3760, 1714, 3759, 3761, 3762, 3763, 3758, 1804, 1803, 1713, 1802), (2, 1): (1699, 1698, 1697, 1700, 4088, 4089, 4090, 4091, 4092, 1696), (2, 2): (3251, 3250, 3249, 1681, 2582, 3252, 3253, 2581, 2583, 3248, 2584), (2, 3): (2478, 2477, 2476, 2479, 3658, 2480, 2481), (2, 4): (2598, 2597, 2596, 3598, 2599, 3524, 2600, 4093), (2, 5): (3777, 2359, 3702, 3778, 3779, 2360, 2361), (2, 6): (3838, 3837, 3836, 2419, 2390, 2299, 2298, 2525, 2089, 2090), (2, 7): (3686, 3687, 3688, 3689, 2118, 2119, 2120, 3580, 2088, 2121), (2, 8): (3536, 3535, 3534, 3538, 3537, 3539, 1682, 2658, 2659, 1683), (3, 0): (1999, 2000, 1998, 1997, 3685, 3684, 2001), (3, 1): (1894, 1893, 1892, 1891, 3909, 1895), (3, 2): (3328, 3327, 3326, 3325, 3324, 2461, 3329, 2101), (3, 3): (3297, 3296, 3299, 3298, 3295), (3, 4): (3913, 2614, 2613, 2612, 2611, 3912, 3914, 3911, 3910), (3, 5): (2570, 2569, 3943, 3942, 3941, 2568, 2567, 3940, 3939), (3, 6): (2523, 3718, 2524, 2522, 3719, 3717, 3716, 3715, 2284), (3, 7): (2194, 2193, 2192, 3641, 3640, 2195), (3, 8): (2029, 3805, 2028, 2027, 2030, 3804), (4, 0): (1744, 1745, 1743, 1742, 3849, 1805, 4029, 1741), (4, 1): (1863, 1862, 1861, 1864, 1865, 3639), (4, 2): (3341, 3342, 2056, 3340, 3343, 3339), (4, 3): (3279, 3281, 3282, 3283, 3280, 3284), (4, 4): (4033, 2495, 4032, 2494, 2493, 2492, 2491, 4031, 4030), (4, 5): (3988, 2540, 2539, 2537, 3987, 3986, 3985, 3984, 2538), (4, 6): (2345, 2344, 2342, 3897, 2343, 2346, 3896, 3895, 3894), (4, 7): (2150, 3851, 2149, 2148, 4046, 2147, 2146, 4045, 4044), (4, 8): (1968, 3520, 3519, 1969, 1970, 1967, 1966), (5, 0): (3444, 1712, 3445, 3446, 3447, 3443), (5, 1): (3400, 3399, 1982, 3401, 3402, 1981, 3398), (5, 2): (3310, 3312, 1951, 3309, 3311, 3313, 1652), (5, 3): (2418, 2417, 2416, 3732, 3733), (5, 4): (2432, 2433, 2434, 2431, 3448, 2435), (5, 5): (3581, 2207, 2208, 3582, 3583, 2223), (5, 6): (2313, 2314, 3657, 3656, 2315, 2312, 2311, 3207, 2430), (5, 7): (3566, 3568, 2267, 3565, 3567, 3564, 3563, 1653), (5, 8): (3491, 3492, 3490, 2117, 3493, 2013, 3489, 2116), (6, 0): (1940, 1939, 1938, 3700, 3699, 4015, 4016, 1937), (6, 1): (4104, 1820, 1819, 1821, 1818, 1817, 1816, 4105, 4106, 4107), (6, 2): (3386, 2282, 3387, 3385, 3384, 2266, 3388, 3383, 3389), (6, 3): (3356, 3357, 3358, 3355, 3359, 2012, 3354), (6, 4): (3673, 3674, 3672, 2641, 2209, 2210, 2211, 2643, 2642), (6, 5): (4047, 2285, 2225, 2224, 2270, 4048, 4049, 2690, 2689, 2688, 2687), (6, 6): (2179, 2180, 2178, 2177, 3776, 3701, 2181), (6, 7): (2164, 2163, 2165, 3671, 2162, 2166, 4001, 3521), (6, 8): (1729, 3865, 1728, 1727, 1726, 3864, 3866, 3867), (7, 0): (1787, 1788, 1789, 3669, 1790, 1786, 3189, 3190, 3191, 3192), (7, 1): (1771, 3234, 3235, 1772, 1773, 1774, 3236, 3237, 3233), (7, 2): (3265, 3266, 2011, 3267, 3268, 3264, 2356, 3263, 2326), (7, 3): (3222, 3223, 3221, 3220, 2401, 2371), (7, 4): (2388, 3612, 3613, 2389, 2387, 2386, 3614), (7, 5): (2237, 2238, 2239, 3522, 3523, 2240, 2236, 2235), (7, 6): (2253, 2254, 2255, 2252, 2251, 3731, 3807, 3808, 2250), (7, 7): (2072, 2073, 3611, 3730, 2071, 2074, 2075, 4076, 4077, 2070), (7, 8): (3461, 3462, 2087, 2086, 3460, 3459), (8, 0): (1909, 1908, 1907, 3924, 1910, 1775, 1911), (8, 1): (1850, 1849, 1848, 1851, 1847, 1846, 3850, 4149), (8, 2): (3506, 3507, 2297, 2296, 3505, 3508, 3504, 3509), (8, 3): (3432, 2357, 3433, 3429, 3434, 3431, 3430, 2552, 2551), (8, 4): (2511, 2510, 2509, 2508, 2507, 4153, 4152, 4151, 4150, 2506), (8, 5): (3928, 2555, 2554, 2553, 3925, 3927, 3926, 2645, 1985, 3463), (8, 6): (2329, 2328, 2327, 3954, 2330, 3957, 3956, 2331, 3955, 3958), (8, 7): (3626, 3628, 3627, 2358, 3625, 3624), (8, 8): (1954, 1955, 1953, 1952, 3670, 4000, 3999, 1986)}
    # 9 cities - fixed_embedding = {(0, 0): (1354, 1355, 1356, 1353, 1352, 3847), (0, 1): (1115, 1144, 3830, 1143, 3666, 3667, 3829, 1142), (0, 2): (3575, 3576, 3577, 1188, 1128, 3574, 3573), (0, 3): (3515, 3514, 3516, 3513, 3517), (0, 4): (3485, 3484, 3486, 3487, 992, 3483), (0, 5): (903, 904, 905, 3499, 3500, 3501, 3502, 906, 3498, 907), (0, 6): (4237, 1312, 1311, 1310, 1309, 1308, 4238, 4236, 4235, 4234), (0, 7): (1368, 3771, 3772, 1564, 1565, 1566, 1369, 1367), (0, 8): (1460, 1458, 3952, 1459, 1461, 1462), (1, 0): (3966, 3965, 3964, 3967, 1190, 1191), (1, 1): (3800, 995, 1099, 3799, 3802, 3801, 1189, 994), (1, 2): (963, 964, 962, 3619, 3618, 965), (1, 3): (935, 3934, 3933, 934, 933, 936, 932), (1, 4): (844, 843, 842, 845, 846, 4098, 3664), (1, 5): (920, 3889, 919, 918, 921, 3888, 922), (1, 6): (3472, 3471, 3470, 1517, 3469, 1518, 1519, 1520, 1521, 3468), (1, 7): (4071, 1221, 4072, 4073, 4070, 4069, 1222), (1, 8): (4100, 4099, 1041, 4101, 4102), (2, 0): (1384, 1383, 3787, 1385, 1386, 3786), (2, 1): (1203, 1204, 1205, 3621, 3622, 3620, 993), (2, 2): (3635, 1023, 3634, 3633, 3636, 3637), (2, 3): (3560, 3559, 3558, 3561, 1187, 3562), (2, 4): (3530, 1007, 3529, 3528, 3531, 3532, 3533), (2, 5): (1473, 3547, 3546, 3545, 3544, 1474, 1475, 1476, 1477, 3543), (2, 6): (4282, 4281, 4280, 4279, 1492, 4278, 1491, 1490, 1489, 1488), (2, 7): (1504, 1505, 1506, 1503, 1502, 3893), (2, 8): (1445, 1446, 1444, 1447, 3907, 1443), (3, 0): (3890, 3892, 1415, 3891, 1416), (3, 1): (3815, 3816, 1249, 3814, 1248, 1039, 1040, 1339), (3, 2): (1157, 1158, 1159, 1160, 947, 3440, 3439, 3438), (3, 3): (3680, 1053, 1052, 3679, 3678, 1054, 1055), (3, 4): (3456, 3455, 3454, 1293, 1294, 1295, 3453), (3, 5): (951, 952, 950, 3949, 3948, 949, 948), (3, 6): (4012, 4011, 4010, 1296, 4009, 4008, 1297, 4013), (3, 7): (3921, 3922, 1250, 1251, 3920, 1252, 4401, 4400, 953), (3, 8): (1400, 1399, 3817, 1401, 3951, 3950), (4, 0): (4041, 4040, 1116, 4042, 4039, 1117), (4, 1): (1009, 1008, 1010, 3904, 3905, 3906, 1011), (4, 2): (978, 977, 3695, 3424, 979, 980, 981, 3423), (4, 3): (889, 888, 3694, 3693, 887, 890, 891, 892), (4, 4): (814, 813, 812, 3903, 815, 816), (4, 5): (3919, 829, 830, 831, 828, 3918, 827), (4, 6): (4130, 4129, 4128, 4131, 966, 4132), (4, 7): (4266, 1371, 4267, 4265, 4264, 4263, 1370), (4, 8): (4221, 4222, 4220, 4219, 996, 4223), (5, 0): (3875, 3876, 3877, 1100, 1101, 3874, 3873, 1102), (5, 1): (3755, 1084, 1085, 3754, 3753, 3756, 1086, 1083), (5, 2): (1070, 4025, 4023, 4024, 1069, 1068, 3935, 1071), (5, 3): (724, 723, 722, 3663, 725, 726), (5, 4): (709, 710, 708, 711, 707), (5, 5): (755, 756, 754, 753, 3813, 757), (5, 6): (4115, 4114, 4113, 4116, 4117), (5, 7): (4161, 4162, 4163, 1581, 4160, 4159, 4158, 1582), (5, 8): (4085, 4083, 4086, 1056, 4087, 4084, 1057), (6, 0): (1324, 1325, 1323, 1326, 3831, 3832), (6, 1): (1220, 3861, 3860, 1219, 3862, 1218, 1217, 3410, 3863), (6, 2): (3650, 1038, 3649, 3651, 3652, 3648, 3653), (6, 3): (3590, 3589, 3588, 1113, 3591, 3592), (6, 4): (3605, 3606, 3604, 3607, 3603, 1098, 3608), (6, 5): (3846, 3845, 3844, 1280, 1281, 1282, 1279, 1278, 3843), (6, 6): (4253, 1536, 4252, 4251, 4250, 4249, 1535, 1534, 1533), (6, 7): (1430, 1431, 4027, 1429, 1428, 1432, 4028, 4026), (6, 8): (1265, 1264, 3936, 3937, 1266, 1267, 1263), (7, 0): (3980, 1145, 1146, 3981, 3979, 3978, 3982), (7, 1): (3724, 1024, 1025, 1026, 3725, 3723, 3726, 1114), (7, 2): (3769, 3770, 874, 873, 3784, 3783, 875, 876), (7, 3): (859, 3859, 858, 860, 861, 3768, 857), (7, 4): (694, 695, 693, 692, 696), (7, 5): (799, 800, 801, 3798, 798, 797), (7, 6): (4191, 4192, 4193, 1551, 4190, 4189, 4188), (7, 7): (4176, 4177, 4175, 4173, 4174, 1161, 4178), (7, 8): (4145, 4144, 4148, 4146, 4147, 4143), (8, 0): (1129, 1130, 3738, 1131, 3740, 3739, 3696, 3697, 1132), (8, 1): (3741, 3711, 1234, 1235, 1236, 3710, 3709, 1233), (8, 2): (1172, 1173, 1174, 3395, 3394, 3393, 1175, 767), (8, 3): (739, 738, 737, 3708, 4203, 740, 741, 4053, 666), (8, 4): (770, 769, 768, 3963, 771, 772), (8, 5): (784, 785, 786, 3828, 783, 782), (8, 6): (4206, 4205, 4204, 4207, 1176, 4208), (8, 7): (1340, 3996, 3997, 3995, 1341, 3994, 3993, 3998), (8, 8): (4056, 4055, 4054, 1206, 4057, 4058)}

    if fixed_embedding != None:
        print("Using fixed embedding :", fixed_embedding)
        min_chain_length    = min(len(chain) for chain in fixed_embedding.values())
        max_chain_length    = max(len(chain) for chain in fixed_embedding.values())
        print("Min chain length      :", min_chain_length)
        print("Max chain length      :", max_chain_length)
        
    if fixed_embedding == None:
        print("Start embedding search...")
        # Creates a Pegasus fixed embedding from the QBM problem using minorminer 
        random_seed          = None
        # Maximum number of failed iterations to improve the current solution, where each iteration attempts to find an embedding
        max_no_improvement   = 25
        # Number of restart attempts before the algorithm stops.
        tries                = 25
        # Maximum number of failed iterations to improve chain lengths in the current solution
        # Each iteration attempts to find an embedding for each variable of S such that it is adjacent to all its neighbours.
        chainlength_patience = 100
        
        t0_embed = time.perf_counter()
        fixed_embedding = minorminer.find_embedding(Q.keys(), qpu.edgelist, 
                                                    random_seed          = random_seed, 
                                                    max_no_improvement   = max_no_improvement, 
                                                    tries                = tries, 
                                                    chainlength_patience = chainlength_patience, 
                                                    verbose              = 2)
        t1_embed = time.perf_counter()

        min_chain_length    = min(len(chain) for chain in fixed_embedding.values())
        max_chain_length    = max(len(chain) for chain in fixed_embedding.values())
        print("Fixed embedding: \n", fixed_embedding)
        print("Min chain length           :", min_chain_length)
        print("Max chain length           :", max_chain_length)
        print("Embedding computation time : ", np.round(t1_embed-t0_embed, 4),"\n")

    ''' Choose a Dwave Sampler (i.e. QPU sampler to run in production) '''

    t0 = time.perf_counter()
    # sampler  = EmbeddingComposite(qpu)
    # sampler  = LazyFixedEmbeddingComposite(qpu, find_embedding = find_embedding, embedding_parameters = None)
    # sampler  = FixedEmbeddingComposite(qpu, fixed_embedding)
    sampler    = KerberosSampler()
    
    # Old version sampling from qubo model directly, specifying required chain_strength 
    # sampleset = sampler.sample_qubo(qubo, num_reads = num_samples, annealing_time = annealing_time, chain_strength = chain_strength)

  # sampleset = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time)
  # sampleset = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time, chain_strength = chain_strength)
  # sampleset = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time, chain_strength = scaled)
  # df_sampleset = sampleset.to_pandas_dataframe()

    print("Calling Kerberos...")
    response = sampler.sample_qubo(Q,
                                    max_iter            = 100, 
                                    max_time            = None,
                                    convergence         = 3,
                                    energy_threshold    = None,
                                    sa_reads            = 1, 
                                    sa_sweeps           = 10000, 
                                    tabu_timeout        = 500,
                                    qpu_reads           = 100, 
                                    qpu_sampler         = DWaveSampler(), 
                                    qpu_params          = None,
                                    max_subproblem_size = 50)
    t1 = time.perf_counter()

    for datum in response.data(['sample', 'energy', 'num_occurrences']):
        print(datum.sample, "Energy: ", np.round(datum.energy, 4) , "Occurrences: ", datum.num_occurrences)
        
        # Decode quantum qubo route from sample
        points_order = []
        binary_state = []
        for key, value in datum.sample.items():
            binary_state.append(value)
   
        number_of_points = int(np.sqrt(len(binary_state)))
        for p in range(number_of_points):
            for j in range(number_of_points):
                if binary_state[(number_of_points) * p + j] == 1:
                    points_order.append(j)
                    
        print("decoded route :", points_order)
        
        if is_hamiltonian_path(G_embedding, points_order) and len(points_order) == Nb_cities:
            
            if points_order[0] != 0:
                index        = points_order.index(0)
                points_order = points_order[index:] + points_order[:index]
                
            print("shifted decoded route :", points_order)                
            named_route = list()
            for i in points_order:
                named_route.append(city_names[i])
            # named_route.append(city_names[0])    
            print("named route   :", named_route)

    '''
    *** Show solution results and compute metrics ***
    '''
    dwave.inspector.show(sampleset) 
    
    problem_id          = sampleset.info['problem_id']
    embedding_qubo      = (sampleset.info["embedding_context"])["embedding"]
    max_chain_length    = max(len(chain) for chain in embedding_qubo.values())
    chain_strength_m    = sampleset.info['embedding_context']['chain_strength']       
    N_logical_variables = len(embedding_qubo.keys())
    N_physical_qubits   = sum(len(chain) for chain in embedding_qubo.values())
    
    print("Problem Id                  :", problem_id)  
    print("Embedding                   :", embedding_qubo)
    print("Max chain length            :", max_chain_length)
    print("Chain strength              :", np.round(chain_strength_m,4))
    print("Number of logical variables :", N_logical_variables)
    print("Number of physical qubits   :", N_physical_qubits)
    print("\nTimings :")
    print("qpu sampling time             :", sampleset.info["timing"]['qpu_sampling_time'])
    print("qpu anneal time per sample    :", sampleset.info["timing"]['qpu_anneal_time_per_sample'])
    print("qpu readout time per sample   :", sampleset.info["timing"]['qpu_readout_time_per_sample'])
    print("qpu access time               :", sampleset.info["timing"]['qpu_access_time'])
    print("qpu access overhead time      :", sampleset.info["timing"]['qpu_access_overhead_time'])
    print("qpu programming time          :", sampleset.info["timing"]['qpu_programming_time'])
    print("qpu delay time per sample     :", sampleset.info["timing"]['qpu_delay_time_per_sample'])
    print("post processing overhead time :", sampleset.info["timing"]['post_processing_overhead_time'])
    print("total post processing time    :", sampleset.info["timing"]['total_post_processing_time'])


    Q_correct_solutions = list()
    Total_correc        = 0
    count               = 0
    have_solution       = False
    cycles_BQM          = []
    best_BQM_score      = np.inf
    
    for e in sampleset.data(sorted_by='energy', sample_dict_cast = False):
                
        # Decode quantum qubo route from sample    
        route_qubo_quantum = [None]*(len(G_embedding))
        for (city, step), val in e.sample.items():
            if val:
                route_qubo_quantum[step] = city
        # print("route_qubo_quantum :", route_qubo_quantum)

        # Reorder cities to have starting city first    
        if Seed_city is not None and route_qubo_quantum[0] != Seed_city and Seed_city in route_qubo_quantum:
            # rotate to put the starting city in front
            idx                = route_qubo_quantum.index(Seed_city)
            route_qubo_quantum = route_qubo_quantum[idx:] + route_qubo_quantum[:idx]
            # print("shifted route_qubo_quantum :", route_qubo_quantum)   
        
        # Check if solution is valid and save the best one (assumed to be the first valid solution of lowest energy)
        if is_hamiltonian_path(G_embedding, route_qubo_quantum):
            # print("Valid solution found!")
            sample_best               = e.sample
            energy_best               = e.energy
            num_cocurrences_best      = e.num_occurrences
            chain_break_fraction_best = e.chain_break_fraction
            
            completed_route_qubo_quantum = copy.deepcopy(route_qubo_quantum)
            completed_route_qubo_quantum.append(Seed_city)
            score_BQM = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == completed_route_qubo_quantum).values]).astype(float)

            if len(score_BQM) == 0:        
                completed_cycle = copy.deepcopy(route_qubo_quantum)
                completed_cycle.append(Seed_city)
                completed_reversed_cycle = list(reversed(completed_cycle))
                                
                # shift cycle by 1 element if start and end cities are the same
                if completed_reversed_cycle[0] == Seed_city and completed_reversed_cycle[1] == Seed_city:
                    completed_reversed_cycle = completed_reversed_cycle[1:] + completed_reversed_cycle[:1]
                
                # print("shifted completed_reversed_cycle :", completed_reversed_cycle)   
                    
                score_BQM          = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == completed_reversed_cycle).values]).astype(float)    
                if len(score_BQM) == 0:
                    print("reversed BQM cycle still not found! \n")
                else:
                    unscaled_score_BQM = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == completed_reversed_cycle).values]).astype(float)
                    rank_score_BQM     = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == completed_reversed_cycle).values]).astype(int)
                    named_route = list()                
                    for i in completed_reversed_cycle[:-1]:
                        named_route.append(city_names[i])
                    named_route.append(city_names[0])           
                    cycles_BQM.append({"cycle": completed_reversed_cycle, "score BQM": score_BQM[0]})       
            else:
                # print("BQM cycle found!")
                route_qubo_quantum.append(Seed_city)
                unscaled_score_BQM = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == route_qubo_quantum).values]).astype(float)
                rank_score_BQM     = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == route_qubo_quantum).values]).astype(int)
                
                named_route = list()
                for i in route_qubo_quantum[:-1]:
                    named_route.append(city_names[i])
                
                cycles_BQM.append({"cycle": route_qubo_quantum, "score BQM": score_BQM[0]})       

            # print("route_qubo_quantum :", cycles_BQM[-1]['cycle'], "energie :", e.energy, "named route :", named_route,"\n")
            
            for i in range(e.num_occurrences):
                Q_sol = {'route': cycles_BQM[-1]['cycle'], 'named route': named_route, 'score_BQM': float(score_BQM), 'energy': e.energy ,"chain_break_fraction": e.chain_break_fraction}
                Q_correct_solutions.append(Q_sol)
            Total_correc += e.num_occurrences
            count += 1

            if not have_solution:
                have_solution                   = True
                lowest_nrj_BQM_score            = float(score_BQM)
                lowest_nrj_unscaled_score_BQM   = float(unscaled_score_BQM)
                lowest_nrj_rank_score_BQM       = int(rank_score_BQM)
                solution_lowest_nrj             = cycles_BQM[-1]['cycle']
                sample_lowest_nrj               = e.sample
                energy_lowest_nrj               = e.energy
                num_cocurrences_lowest_nrj      = e.num_occurrences
                chain_break_fraction_lowest_nrj = e.chain_break_fraction

            if float(score_BQM) <= best_BQM_score:
                best_BQM_score            = float(score_BQM)
                best_unscaled_score_BQM   = float(unscaled_score_BQM)
                best_rank_score_BQM       = int(rank_score_BQM)
                solution_best             = cycles_BQM[-1]['cycle']
                sample_best               = e.sample
                energy_best               = e.energy
                num_cocurrences_best      = e.num_occurrences
                chain_break_fraction_best = e.chain_break_fraction

    Percentage_correct_routes = 0.0

    if len(Q_correct_solutions) != 0:
        df_Q_correct_solutions = pd.DataFrame.from_dict(Q_correct_solutions)
        df_Q_spectrum_energie  = df_Q_correct_solutions.groupby([df_Q_correct_solutions['route'].map(tuple),'energy'])['route'].count().to_frame(name = 'count').reset_index()
        df_Q_spectrum_type     = df_Q_correct_solutions.groupby([df_Q_correct_solutions['route'].map(tuple)])['route'].count().to_frame(name = 'count').reset_index()

        Percentage_correct_routes = round((df_Q_spectrum_type[['count']].values.sum()/num_samples)*100., 3)
        print("Percentage valid routes: ", Percentage_correct_routes)
        print("Nb unique  valid routes: ", len(df_Q_spectrum_type))

    df_timing = pd.DataFrame(sampleset.info["timing"], index=[0]).T
    df_timing = df_timing.sort_values(by=[0], ascending = False)
    df_timing.rename(columns = {0:'Duration (μs)'}, inplace = True)

    '''
    Save quantum results to file
    '''
    name_quantum     = "solution_quantum-cities_" + str(Nb_cities)\
                     + "_seed_"     + str(Seed_city) \
                     + "_lag_"      + str(np.round(lagrange_multiplier,2)) \
                     + "_chain_"    + str(np.round(chain_strength,2)) \
                     + "_samples_"  + str(num_samples) \
                     + "_AT_"       + str(annealing_time) \
                     + "_ok_"       + str(Percentage_correct_routes)
                         

    name_QBM_spec    = "solution_QBM_spectrum_cities_" + str(Nb_cities)\
                     + "_seed_"     + str(Seed_city) \
                     + "_lag_"      + str(np.round(lagrange_multiplier,2)) \
                     + "_chain_"    + str(np.round(chain_strength,2)) \
                     + "_samples_"  + str(num_samples) \
                     + "_AT_"       + str(annealing_time) \
                     + "_ok_"       + str(Percentage_correct_routes)
    
    name_QBM_map     = "solution_QBM_map_cities_" + str(Nb_cities)\
                     + "_seed_"     + str(Seed_city) \
                     + "_lag_"      + str(np.round(lagrange_multiplier,2)) \
                     + "_chain_"    + str(np.round(chain_strength,2)) \
                     + "_samples_"  + str(num_samples) \
                     + "_AT_"       + str(annealing_time) \
                     + "_ok_"       + str(Percentage_correct_routes)

# Define output files for Kerberos solver experiments
    name_Kerberos_spec = "solution_QBM_spectrum_cities_" + str(Nb_cities)\
                       + "_seed_"     + str(Seed_city)
    name_Kerberos_map  = "solution_Kerberos_map_cities_" + str(Nb_cities)\
                       + "_seed_" + str(Seed_city)

    out_file_Quantum       = path + name_quantum       + ext1
    out_file_QBM_spec      = path + name_QBM_spec      + ext2
    out_file_QBM_map       = path + name_QBM_map       + ext2
    out_file_Kerberos_map  = path + name_Kerberos_map  + ext2
    out_file_Kerberos_spec = path + name_Kerberos_spec + ext2

    with open(out_file_Quantum, 'w') as f:  
        f.write(f"Problem Id: {problem_id}\n") # does not depend on sample
        chain_break_fraction = np.sum(sampleset.record.chain_break_fraction)/num_samples
        
        if have_solution:
            f.write("Solution best:\n")
            f.write(f"{solution_best}\n")
            f.write(f"score quantum best              : {best_BQM_score}\n")
            f.write(f"sample best                     : \n {sample_best}\n")
            f.write(f"rank best                       : {best_rank_score_BQM}\n")
            f.write(f"energy best                     : {energy_best}\n")
            f.write(f"% error best                    : {np.round(((best_BQM_score - best_score_BF) /best_score_BF * 100.), 2)}\n")
            f.write(f"num occurrences best            : {num_cocurrences_best}\n")
            f.write(f"chain break fraction best       : {chain_break_fraction_best}\n\n")
            f.write("Solution lowest NRJ:\n")
            f.write(f"{solution_lowest_nrj}\n")
            f.write(f"score quantum lowest NRJ        : {lowest_nrj_BQM_score}\n")
            f.write(f"sample lowest NRJ               : \n {sample_lowest_nrj}\n")
            f.write(f"rank lowest NRJ                 : {lowest_nrj_rank_score_BQM}\n")
            f.write(f"energy lowest NRJ               : {energy_lowest_nrj}\n")
            f.write(f"% error lowest NRJ              : {np.round(((lowest_nrj_BQM_score - best_score_BF) /best_score_BF * 100.), 2)}\n")
            f.write(f"num occurrences lowest NRJ      : {num_cocurrences_lowest_nrj}\n")
            f.write(f"chain break fraction lowest NRJ : {chain_break_fraction_lowest_nrj}\n\n")
            f.write(f"number of logical variables     : {N_logical_variables}\n")
            f.write(f"number of physical qubits       : {N_physical_qubits}\n")
            f.write(f"max chain length                : {max_chain_length}\n")
            f.write(f"max graph chain strength        : {max_chain_strength}\n")
            f.write(f"relative chain strength         : {RCS}\n")
            f.write(f"chain strength                  : {np.round(chain_strength,5)}\n")
            f.write(f"chain break fraction            : {np.round(chain_break_fraction,5)}\n")
            f.write(f"annealing time                  : {annealing_time}\n")
            f.write(f"relative Lagrange multiplier    : {RLM}\n")
            f.write(f"lagrange multiplier             : {np.round(lagrange_multiplier,5)}\n")
            f.write(f"percentage correct routes       : {Percentage_correct_routes}\n")
            f.write(f"time                            : {t1-t0:0.4f} s\n")
        
        if not have_solution:
            f.write("did not find any solution\n")
            f.write(f"chain break fraction  : {chain_break_fraction}\n")

'''
# Get locations Lon/Lat coordinates
'''

geolocator = Nominatim(user_agent="MyApp")
lats = list()
lons = list()
for name in city_names:
    location = geolocator.geocode(name)
    lats.append(location.latitude)
    lons.append(location.longitude)

'''
# Donwload high-res map of the area
'''

m = Basemap(projection ='merc',
            llcrnrlon  = min_lon,
            llcrnrlat  = min_lat,
            urcrnrlon  = max_lon,
            urcrnrlat  = max_lat,
            epsg       = 3857,
            resolution = 'f',
            lon_0      = 0.0,
            lat_0      = 0.0)

parallels = np.arange(round(min_lat), round(max_lat), .5)
meridians = np.arange(round(min_lon), round(max_lon), .5)

# convert lat and lon to map projection
mx, my = m(lons, lats)
G      = nx.complete_graph(n_nodes)
G.name = "Graph of " + str(n_nodes) + " french cities"

mapping = dict()
rename  = dict()
for i in city_index:
    mapping.update({i: "$" + str(city_names[i]) + "$"})
    rename.update( {i: str(city_names[i])})

# Calculating the distances between the nodes as edge's weight.
for i in city_index:
    for j in range(i+1, n_nodes):
        G.add_edge(i, j, weight=int(df_km[i][j]))

pos = {}
for i, name in enumerate(city_names):
    pos[i] = (mx[i], my[i])

''' Plot map of optimized routes for each approach '''
fig, ax = plt.subplots()
fig.set_size_inches(20, 20)

# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
# m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service = 'World_Topo_Map', xpixels = 1000, verbose=True)
m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', verbose = False)

m.drawrivers(    linewidth=1.0, linestyle='solid', color='seagreen' , antialiased=1, ax=None, zorder = 1)
m.drawcoastlines(linewidth=1.0, linestyle='solid', color='steelblue', antialiased=1, ax=None, zorder = 2)
m.drawcountries( linewidth=1.0, linestyle='solid', color='black'    , antialiased=1, ax=None, zorder = 3)

# Draw parallels.
m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize = 10, zorder = 4)
# Draw meridians
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize = 10, zorder = 5)

# nodes
nx.draw_networkx_nodes(G, pos, node_size = 50, node_color = "crimson", label = mapping)
# edges
nx.draw_networkx_edges(G, pos, edge_color = "silver", width = 2.0, alpha = 0.25)

# node labels
pos_higher = {}
y_off      = round(np.mean(my) * 0.02)
for k, v in pos.items():
    pos_higher[k] = (v[0], v[1] + y_off)
    
nx.draw_networkx_labels(G, pos_higher, labels = mapping, font_size = 20, font_family = "sans-serif", font_color = "crimson")

# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size = 8, font_family = "sans-serif", font_color = "black", alpha = 1.0)

''' Compute problem's solution using 'Simulated Annealing' approximate algorithm '''
# For best result
# Temperature = 100
# N_inner     = 500
# Alpha       = 0.005

t4 = time.perf_counter()
t5 = time.perf_counter()
cycles_SA = []
repeat_SA_count = 0

if COMPUTE_QUANTUM:
    Temperature = 100
    N_outer     = 10
    N_inner     = 100
    Alpha       = 0.01

    SA_timings             = list()
    SA_scores              = list()
    best_score_SA          = np.inf
    best_route_SA          = list()
    best_unscaled_score_SA = np.inf
    best_rank_score_SA     = np.inf

    while (t5-t4) <= (t1-t0):
        seed      = int(random.random()*10000)
        cycle     = nx_app.simulated_annealing_tsp(G_embedding, init_cycle = "greedy", weight="weight", temp = Temperature, move='1-1', source = Seed_city, max_iterations = N_outer, N_inner = N_inner, alpha = Alpha, seed=seed)    
        score_SA  = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(float)

        if len(score_SA) == 0:        
            reversed_cycle     = list(reversed(cycle))
            score_SA           = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(float)    
            if len(score_SA) == 0:
                print("reversed SA cycle still not found! \n")
            else:
                unscaled_score_SA  = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(float)
                rank_score_SA      = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(int)
                named_route = list()                
                for i in reversed_cycle[:-1]:
                    named_route.append(city_names[i])
                named_route.append(city_names[0])           
                cycles_SA.append({"cycle": reversed_cycle, "score SA": score_SA[0]})       

        else:
            unscaled_score_SA  = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(float)
            rank_score_SA      = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(int)
            named_route = list()
            for i in cycle[:-1]:
                named_route.append(city_names[i])
            named_route.append(city_names[0])
            cycles_SA.append({"cycle": cycle, "score SA": score_SA[0]})       

        SA_scores.append(score_SA)
        if score_SA <= best_score_SA:
            best_score_SA          = score_SA
            best_route_SA          = named_route
            best_unscaled_score_SA = unscaled_score_SA
            best_rank_score_SA     = rank_score_SA  
        # print("Simulated Annealing computation time : ", t5-t4, " (", repeat, " repeats)")
    
        repeat_SA_count += 1
        t5 = time.perf_counter()
        SA_timings.append(t5-t4)
        
    SA_timings = np.array(SA_timings)
    SA_scores  = np.array(SA_scores)
else:
    Temperature = 100.0
    N_outer     = 10
    N_inner     = 100
    Alpha       = 0.01

    '''
    *** Repeat SA experience N times for statistics
    '''
    N_exp   = 100
    SA_timings             = list()
    SA_scores              = list()
    best_score_SA          = np.inf
    best_route_SA          = list()
    best_unscaled_score_SA = np.inf
    best_rank_score_SA     = np.inf
        
    for repeat in range(N_exp):
        t4                 = time.perf_counter()
        seed               = int(random.random()*10000)
        cycle              = nx_app.simulated_annealing_tsp(G_embedding, init_cycle = "greedy", weight="weight", temp = Temperature, move='1-1', source = Seed_city, max_iterations = N_outer, N_inner = N_inner, alpha = Alpha, seed=seed)    
        # cycle              = nx_app.simulated_annealing_tsp(G_embedding, init_cycle = init_cycle, weight="weight", temp = Temperature, move='1-1', source = Seed_city, max_iterations = N_outer, N_inner = N_inner, alpha = Alpha, seed=seed)
        # cycle              = nx_app.simulated_annealing_tsp(G_embedding, init_cycle = init_cycle, weight="weight", temp = Temperature, move="1-0", source = Seed_city, max_iterations = N_outer, N_inner = N_inner, alpha = Alpha, seed=seed) 
        t5                 = time.perf_counter()

        score_SA           = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(float)

        if len(score_SA) == 0:        
            reversed_cycle     = list(reversed(cycle))
            score_SA           = np.array((spectrum_brute_force["score BF"])[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(float)    
            if len(score_SA) == 0:
                print("reversed SA cycle still not found! \n")
            else:
                unscaled_score_SA  = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(float)
                rank_score_SA      = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == reversed_cycle).values]).astype(int)
                named_route = list()                
                for i in reversed_cycle[:-1]:
                    named_route.append(city_names[i])
                named_route.append(city_names[0])
                
                cycles_SA.append({"cycle": reversed_cycle, "score SA": score_SA[0]})       
                # print("repeat: *", repeat, " cycle:", reversed_cycle, "score SA:", score_SA[0])
        else:
            unscaled_score_SA  = np.array((spectrum_brute_force["unscaled_score BF"])[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(float)
            rank_score_SA      = np.array((spectrum_brute_force.index)[spectrum_brute_force.route.apply(lambda x: x == cycle).values]).astype(int)
            named_route = list()
            for i in cycle[:-1]:
                named_route.append(city_names[i])
            named_route.append(city_names[0])
            cycles_SA.append({"cycle": cycle, "score SA": score_SA[0]})       
            # print("repeat  :", repeat, " cycle:", cycle, "score SA:", score_SA[0])
                
        # print(named_route)
        repeat_SA_count += 1
        SA_timings.append(t5-t4)
        SA_scores.append(score_SA)
        if score_SA <= best_score_SA:
            best_score_SA          = score_SA
            best_route_SA          = named_route
            best_unscaled_score_SA = unscaled_score_SA
            best_rank_score_SA     = rank_score_SA  
        # print("Simulated Annealing computation time : ", t5-t4, " (", repeat, " repeats)")
    
    SA_timings = np.array(SA_timings)
    SA_scores  = np.array(SA_scores)

cycles_SA_pd  = pd.DataFrame(cycles_SA)
best_SA       = cycles_SA_pd.iloc[cycles_SA_pd["score SA"].idxmin()]
best_SA_score = best_SA["score SA"]
best_SA_cycle = best_SA["cycle"]
spectrum_SA   = pd.DataFrame (cycles_SA, columns = ["cycle","score SA"])

# ''' Reverse Simulated annealing route if opposite to quantum route '''
# if COMPUTE_QUANTUM and have_solution:
#     if cycle[1] != quantum_route[1]:
#         cycle.reverse()

# ''' Reverse Brute Force route if opposite to quantum route '''
# if COMPUTE_QUANTUM and have_solution:
#     if BF_route[1] != quantum_route[1]:
#         BF_route.reverse()

''' If no quantum solution reverse Simulated annealing route if opposite to Brute Force route '''
if COMPUTE_QUANTUM or not have_solution:
    if cycle[1] != BF_route[1]:
        cycle.reverse()

if COMPUTE_QUANTUM and have_solution:
    
    print("\nBQM quantum solver lowest NRJ result       :", solution_lowest_nrj)
    print("Lowest NRJ quantum route score is        :", lowest_nrj_BQM_score)
    for i in solution_lowest_nrj[:-1]:
        print(city_names[i], " -> ", end = '')
    print(city_names[0])
    print("The lowest NRJ QBM unscaled route score is : ", float(lowest_nrj_unscaled_score_BQM))
    print("The lowest NRJ QBM rank route score is     : ", int(lowest_nrj_rank_score_BQM))
    print("Error lowest NRJ QBM route                 : ", np.round(((lowest_nrj_BQM_score - best_score_BF) /best_score_BF * 100.), 2),"%")
    print("Chain break fraction lowest NRJ solution   : ", np.round(chain_break_fraction_lowest_nrj, 5),"\n")

    print("BQM quantum solver best result             :", solution_best)
    print("Best quantum route score is : ", best_BQM_score)
    for i in solution_best[:-1]:
        print(city_names[i], " -> ", end = '')
    print(city_names[0])
    print("The best QBM unscaled route score is       : ", float(best_unscaled_score_BQM))
    print("The best QBM rank route score is           : ", int(best_rank_score_BQM))
    print("Error Best QBM route                       : ", np.round(((best_BQM_score - best_score_BF) /best_score_BF * 100.), 2),"%")    
    print("Chain break fraction best solution         : ", np.round(chain_break_fraction_best, 5),"\n")

print("Quantum percentage of correct solutions    : ", Percentage_correct_routes)
print("Chain break fraction global                : ", np.round(chain_break_fraction, 5))
print("Quantum computation time                   : ", np.round(t1-t0, 4),"\n\n")

print("The brute-force route score is         : ", best_score_BF)
for i in BF_route[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0],"\n")

print("Over", repeat_SA_count,"SA repeats     :")
print("The Simulated Annealing route score is : ", best_SA_score)
for i in best_SA_cycle[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0])

print("The best SA unscaled route score is    : ", float(best_unscaled_score_SA))
print("The best SA rank route score is        : ", int(best_rank_score_SA))
print("Error Best SA route                    : ", np.round(((best_SA_score - best_score_BF) /best_score_BF * 100.), 2),"%")    
print("Mean score SA                          : ", np.round(np.mean(SA_scores),4))
print("Std  score SA                          : ", np.round(np.std(SA_scores) ,4))
print("Error Mean SA route                    : ", np.round(((np.mean(SA_scores) - best_score_BF) /best_score_BF * 100.), 2),"%")
print("SA computation time                    : ", np.round((t5-t4),4))
print("Mean itereation computation time       : ", np.round(np.mean(SA_timings),4))
print("Std  itereation computation time       : ", np.round(np.std(SA_timings) ,4))    

# Draw the routes
nx.draw_networkx_nodes(G, pos, nodelist = [0], node_size = 100, node_color = "gold", label = mapping)

edge_list_SA = list(nx.utils.pairwise(best_SA_cycle))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_SA, edge_color = "mediumblue", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.5)

SA_patch = mpatches.Patch(color="mediumblue", label="Simulated Annealing route")

edge_list_BF          = list(nx.utils.pairwise(BF_route))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_BF, edge_color = "green", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.5)

BF_patch = mpatches.Patch(color="green", label="Brute-Force route")

if COMPUTE_QUANTUM and have_solution:
    edge_list_quantum = list(nx.utils.pairwise(solution_best))
    nx.draw_networkx_edges(G, pos, edgelist = edge_list_quantum, edge_color = "red" , arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.5)
    quantum_patch = mpatches.Patch(color="red", label="Quantum route")
    ax.legend(handles=[SA_patch, BF_patch, quantum_patch])
else:
    ax.legend(handles=[SA_patch, BF_patch])

plt.title(G.name)
plt.tight_layout()
plt.show()

normalized_spectrum_SA = pd.DataFrame((spectrum_SA["score SA"] - spec_min) / (spec_max - spec_min))

if COMPUTE_QUANTUM:
    spectrum_Qantum = pd.DataFrame(Q_correct_solutions, columns = ["score_BQM", "route"])
    normalized_spectrum_Qantum = pd.DataFrame((spectrum_Qantum["score_BQM"] - spec_min) / (spec_max - spec_min))

''' Plot spretra difference for each approach '''

plt.show()
# fig.savefig(out_file_QBM_map)
fig.savefig(out_file_Kerberos_map)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

# Binerize the distributions
n_brute_force, bin_edges    = np.histogram(normalized_spectrum_brute_force["score BF"], bins = nb_bin, range=(0., 1.0))
# bin_probability_brute_force = n_brute_force /float(n_brute_force.sum())
bin_probability_brute_force = n_brute_force /float(n_brute_force.max())

n_SA         , _            = np.histogram(normalized_spectrum_SA["score SA"]         , bins = nb_bin, range=(0., 1.0))
bin_probability_SA          = n_SA /float(n_SA.sum())

if COMPUTE_QUANTUM and have_solution:
    n_Qantum     , _         = np.histogram(normalized_spectrum_Qantum["score_BQM"], bins = nb_bin, range=(0., 1.0))
    bin_probability_Qantum   = n_Qantum /float(n_Qantum.sum())

# Get the mid points of every bin
# bin_middles = (bin_edges[1:] + bin_edges[:-1])/2.
bin_middles = (bin_edges[0:-1] + bin_edges[:-1])/2.

# Compute the bin-width
bin_width   = (bin_edges[1]-bin_edges[0]) / 6.

# Make a multiple-histogram of spectra density distributions
ax.bar(bin_middles              , height = bin_probability_brute_force, width = bin_width, alpha = 0.5, color = 'slategrey' , label="Solution space Energy spectrum")
ax.bar(bin_middles + bin_width  , height = bin_probability_SA         , width = bin_width, alpha = 0.5, color = 'mediumblue', label="Simulated Annealing spectrum")

if COMPUTE_QUANTUM and have_solution:
    ax.bar(bin_middles + 2*bin_width, height = bin_probability_Qantum , width = bin_width, alpha = 0.5, color = 'red'   , label="Quantum Annealing spectrum")
    y_limit = np.max([bin_probability_brute_force, bin_probability_SA, bin_probability_Qantum])
else:
    y_limit = np.max([bin_probability_brute_force, bin_probability_SA])
ax.set_xticks(bin_edges)
ax.set_xticklabels(np.round(bin_edges,3), rotation=65)

ax.set_title("Normalized probability distributions of solutions' energy")
ax.set_xlabel("Normalized routes scores", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Probability density"     , labelpad=20, weight='bold', size=12)
ax.set_ylim(0.0, y_limit)

score_BF = trunc((best_score_BF - spec_min) / (spec_max - spec_min), decs = round_digit)
ax.axvspan(score_BF - bin_width/2., score_BF + bin_width/2., ymin = 0, ymax = 1.0, alpha=0.4, color='green', label="Brute-Force best score")

plt.legend(loc='upper right')
plt.tight_layout()
# fig.savefig(out_file_QBM_spec)
fig.savefig(out_file_Kerberos_spec)