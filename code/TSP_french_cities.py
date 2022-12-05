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
from mpl_toolkits.basemap import Basemap as Basemap
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler

from dwave.embedding.chain_strength import scaled
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import dwave.inspector

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

def index(i, j, n):
    if i == j:
        raise ValueError
    elif i > j:
        return index(j, i, n)
    else:
        return int(i*n - (i*(i+1)/2) + j - (i+1))

def build_objective_matrix(M):
    n, _ = M.shape
    # m is the total of binary variables we have to solve for
    # basically given any two nodes, we need to decide if there is an edge connecting them (a binary variable)
    m = int(n*(n-1)/2)
    Q = np.zeros((m,m))
    k = 0
    for i in range(0, n):
        for j in range(i+1, n):
            # M[i,j] + M[j,i] is the cost to travel from i to j (or vice-versa)
            Q[k, k] = (M[i,j] + M[j,i])
            k += 1
    # diagonal matrix of biases
    return Q

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

def build_constraint_matrix(n, bias_value = -5):
    """
     The constraints matrix encodes the constraint that each city (node) is connected to exactly two other cities in the output cycle        
    """
    m          = int(n*(n-1)/2)
    C          = np.zeros((m,m))
    for i in range(0, n):
        # Diagonal terms of C are equal to 2*bias_value
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
                C[ia,ib] += 1.0
    return C

def is_valid_solution(X):
    rows, cols = X.shape
    for i in range(rows):
        count = 0
        for j in range(cols):
            if X[i,j] == 1:
                count += 1
        if not count == 2:
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

Nb_cities = 5
random.seed(0)

cities_keep    = random.sample(city_names, Nb_cities)
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
n_nodes = len(df.columns)
city_index = list(range(0, n_nodes))
df.columns = list(range(0, n_nodes))

df = df.fillna(0)
df = df.astype(int)

''' The objective should be that elements of the qubo matrix are all below the chain strength valu '''

''' Define model's meta-parameters '''
# Scaling factor for the pairwaise cost matrix between nodes

df_km = copy.deepcopy(df)
scaling_factor = 5.0

scaler = MinMaxScaler(feature_range=(0, scaling_factor))
df     = scaler.fit_transform(df)
# df   = df.to_numpy()

# Bias value used to build the constraint matrix
bias_value = -3.5

# Lagrange multiplier for taking into account the constraint matrix
# lagrange_multiplier = 4.0
lagrange_multiplier = np.max(np.abs(df))

# Number of measurements repetition (up to 10000)
num_samples      = 1000
out_file         = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/solution_quantum-"+str(Nb_cities)+"_cities.txt"
out_file_BF      = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/solution_BF-"     +str(Nb_cities)+"_cities.txt"
out_file_BF_all  = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/solution_BF_all-" +str(Nb_cities)+"_cities.txt"
out_file_BF_spec = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/solution_BF_spectrum-" +str(Nb_cities)+"_cities.png"

'''
*** Define objective matrix of travel costs ***
# M is the matrix of pairwise costs (i.e. cost to travel from node i to node j)
# M need not be a symmetric matrix but the diagonal entries are ignored and assumed to be zero
'''
M                   = df
Q                   = build_objective_matrix(M)
Decode              = build_decode(M, city_names)

''' 
*** Encode connectivity constraint in the final cycle ***
# We need to add the constraint that each city is connected to exactly 2 other cities
# We do this using the method of lagrange multipliers where the constraint is absorbed into the objective function
# (this is the hardest part of the problem)
'''
n, _       = M.shape
C          = build_constraint_matrix(n, bias_value)

'''
*** Define the Hamiltonian of the problem ***
'''
qubo = Q + lagrange_multiplier * C

'''
H = HA + HB
HA : Hamiltonian component corresponding to the (directed or undirected) Hamiltonian cycle problem
HB : Hamiltonian component corresponding to the Weighted graph problem
HA = A . term_1 + A term_2 + A term-3
NB = B . term_4
B sould be small enough i.e. 0 < B max(Wuv) < A
so that no solution can favour to violate the constraints of HA

term_1 : constraint that every vertex can only appear once in a cycle      (i.e. all nodes must be visited at most once)
term_2 : constraint that "there must be a jth node in the cyle for each j" (i.e. all nodes need to be visited at least once)
term_3 : if Xu,j and Xv,j+1 are both 1 then there should be an energie penalty if (u,v) is not in the edgeset of the graph
    
In the implementation :
    
M                  <-> Wuv
Q                  <-> HA
C                  <-> HB
lagrange_mutiplier <-> (B/A)
qubo               <-> H
'''

print("Cost matrix M        : \n", M, "\n")
print("Objective matrix Q   : \n", Q, "\n")
print("Lagrange multiplier  : \n", lagrange_multiplier, "\n")
print("Constraints matrix C : \n", C, "\n")
print("QUBO matrix qubo     : \n", qubo, "\n")

'''
*** Chose a DWave sampler ***
# QPU sampler to run in production
'''
sampler = EmbeddingComposite(DWaveSampler())

# Gain more control on the minor embedding process using code below ?

# sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
# linear       = {('a', 'a'): -1, ('b', 'b'): -1, ('c', 'c'): -1}
# quadratic    = {('a', 'b'): 2, ('b', 'c'): 2, ('a', 'c'): 2}
# Q            = {**linear, **quadratic}
# sampleset = sampler_auto.sample_qubo(Q, num_reads = num_samples)

#

'''
*** Run the problem on the QPU recording execution times
'''
have_solution      = False
if COMPUTE_QUANTUM:
    t0        = time.perf_counter()
    sampleset = sampler.sample_qubo(qubo, num_reads=num_samples, chain_strength=scaled)    
    t1        = time.perf_counter()
    
    '''
    *** Show solution results and compute metrics ***
    '''
    print(sampleset)
    # print(sampleset.info.keys())
    # print(sampleset.info["timing"])
    print(sampleset.info["embedding_context"])
    
    dwave.inspector.show(sampleset)    
    problem_id         = sampleset.info['problem_id']
    chain_strength     = sampleset.info['embedding_context']['chain_strength']
    Q_correct_solution = list()
    Total_correc       = 0

    with open(out_file, 'w') as f:
        f.write(f"Problem Id: {problem_id}\n") # does not depend on sample  
        count = 0
        
        for e in sampleset.data(sorted_by='energy', sample_dict_cast = False):
            X = build_solution(e.sample)
            
            if is_valid_solution(X) and not have_solution:
                have_solution             = True
                score_best                = compute_score(M, X)
                solution_best             = X
                sample_best               = e.sample
                energy_best               = e.energy
                num_cocurrences_best      = e.num_occurrences
                chain_break_fraction_best = e.chain_break_fraction
                f.write("Solution:\n")
                f.write(f"{solution_best}\n")
                f.write(f"Score: {score_best}\n")
                f.write(f"{sample_best}\n")
                f.write(f"index: {count}\n")
                f.write(f"energy: {energy_best}\n")
                f.write(f"num_occurrences: {num_cocurrences_best}\n")            
                f.write(f"chain break fraction: {chain_break_fraction_best}\n")
                f.write(f"scaling_factor M: {scaling_factor}\n")
                f.write(f"bias_value     C: {bias_value}\n")
                f.write(f"lagrange_multiplier: {lagrange_multiplier}\n")
               
                # break   # break out of for loop (if only looking for the first correct solution)
            
            if is_valid_solution(X): 
                Q_route = decode_route(X)
                score   = compute_score(M, X)
                for i in range(e.num_occurrences):
                    Q_sol = {'route': Q_route, 'score Quantum': score, 'energie': e.energy, "solution": X, "chain_break_fraction": e.chain_break_fraction}
                    Q_correct_solution.append(Q_sol)
                Total_correc += e.num_occurrences
            count += 1
            
        Percentage_correct_routes = (Total_correc/num_samples)*100.
        
        f.write(f"chain strength: {chain_strength}\n")  # does not depend on sample        
        f.write(f"scaling_factor M: {scaling_factor}\n")
        f.write(f"bias_value     C: {bias_value}\n")
        f.write(f"lagrange_multiplier: {lagrange_multiplier}\n")
        f.write(f"Percentage correct routes : {Percentage_correct_routes}\n")
        f.write(f"Time: {t1-t0:0.4f} s\n")
        
        if not have_solution:
            # https://docs.ocean.dwavesys.com/en/latest/examples/inspector_graph_partitioning.html
            # this is the overall chain break fraction
            chain_break_fraction = np.sum(sampleset.record.chain_break_fraction)/num_samples
            f.write("did not find any solution\n")
            f.write(f"chain break fraction: {chain_break_fraction}\n")



if COMPUTE_QUANTUM and have_solution:
    quantum_route   = decode_route(solution_best)

print("Quantum percentage of correct solutions : ", Percentage_correct_routes)

''' Compute problem's solution by 'brute-force' algorithm '''
k                    = 0
best_matrix          = np.zeros((n,n))
best_score_BF        = np.inf
second_best_score_BF = np.inf
unique_solutions     = []

brute_force_results = list()

t0 = time.perf_counter()
with open(out_file_BF_all, 'w') as f:
    for A in enumerate_all_rings(n):
        # 'multiply' does element-wise multiplication (i.e. indivual edge cost)
        # 'sum' will sum the costs over all edges of the cycle
        score = np.sum(np.multiply(M, A))
        route = decode_route(A)
        f.write("{0} {1}\n".format(k, score))
        f.write("{0} \n".format(route))
        f.write("{0} \n".format([A]))
        temp = {"score BF": score, "solution": [A], "route": route}
        brute_force_results.append(temp)
        k +=1
        if score == best_score_BF:
            unique_solutions.append(A)
        elif score < best_score_BF:
            second_best_score_BF = best_score_BF
            best_score_BF        = score
            best_matrix          = A
            best_route           = route
            unique_solutions     = [A]
        elif score < second_best_score_BF:
            second_best_score_BF = score
t1 = time.perf_counter()

spectrum_brute_force = pd.DataFrame (brute_force_results, columns = ["score BF","solution"])
spec_min = spectrum_brute_force["score BF"].min()
spec_max = spectrum_brute_force["score BF"].max()
normalized_spectrum_brute_force = pd.DataFrame((spectrum_brute_force["score BF"] - spec_min) / (spec_max - spec_min))

spectrum_res = 0.005
round_digit  = len(str(int(1./spectrum_res)))-1
spectrum_min = trunc(normalized_spectrum_brute_force.min(), decs = round_digit)
spectrum_max = trunc(normalized_spectrum_brute_force.max(), decs = round_digit)
nb_bin       = int(np.ceil((spectrum_max - spectrum_min)) / spectrum_res)

with open(out_file_BF, 'w') as f:
    f.write("Best score: {0}\n".format(best_score_BF))
    f.write("Number of distinct solutions: {0}\n".format(len(unique_solutions)))
    f.write("Best route: {0}\n".format(best_route))
    for solution in unique_solutions:
        f.write("{0}\n".format(solution))
    f.write("Second best score: {0}\n".format(second_best_score_BF))
    f.write("Energy difference: {0}\n".format(best_score_BF - second_best_score_BF))
    f.write(f"Time: {t1-t0:0.4f} s\n")

solution = unique_solutions[0]

BF_route = decode_route(solution)

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
fig, ax = plt.subplots()
fig.set_size_inches(20, 20)

m = Basemap(projection ='merc',
            llcrnrlon  = min_lon,
            llcrnrlat  = min_lat,
            urcrnrlon  = max_lon,
            urcrnrlat  = max_lat,
            epsg       = 3857,
            resolution = 'f',
            lon_0      = 0.0,
            lat_0      = 0.0)

# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
# m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service = 'World_Topo_Map', xpixels = 1000, verbose=True)
m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', verbose = False)

m.drawrivers(    linewidth=1.0, linestyle='solid', color='seagreen' , antialiased=1, ax=None, zorder = 1)
m.drawcoastlines(linewidth=1.0, linestyle='solid', color='steelblue', antialiased=1, ax=None, zorder = 2)
m.drawcountries( linewidth=1.0, linestyle='solid', color='black'    , antialiased=1, ax=None, zorder = 3)

# Draw parallels.
parallels = np.arange(round(min_lat), round(max_lat), .5)
m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize = 10, zorder = 4)
# Draw meridians
meridians = np.arange(round(min_lon), round(max_lon), .5)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize = 10, zorder = 5)

# convert lat and lon to map projection
mx, my = m(lons, lats)

G = nx.complete_graph(n_nodes)
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
# cycle = nx_app.christofides(G, weight="weight")
N_repeat_SA = 10
Temp        = 100
N_inner     = 1000
Alpha       = 0.005

cycles_SA = []
for i in range(N_repeat_SA):
    cycle    = nx_app.simulated_annealing_tsp(G, init_cycle = "greedy", weight="weight", temp = Temp, move='1-1', source = 0, max_iterations = 10, N_inner = N_inner, alpha = Alpha, seed=None)
    # cycle    = nx_app.simulated_annealing_tsp(G, init_cycle = "greedy", weight="weight", temp = 100, move='1-1', source = 0, max_iterations = 2, N_inner=10, alpha=0.01, seed=None)

    try:
        best_score_SA = (next(item for item in brute_force_results if item["route"] == cycle))["score BF"]
    except StopIteration:
        cycle.reverse()
        best_score_SA = (next(item for item in brute_force_results if item["route"] == cycle))["score BF"]
    
    # print("Ith :", i, " Simulated Annealing route score : ", best_score_SA)    
    cycles_SA.append({"cycle": cycle, "score SA": best_score_SA})
    
spectrum_SA = pd.DataFrame (cycles_SA, columns = ["cycle","score SA"])


''' Reverse Christofides route if opposite to quantum route '''
if COMPUTE_QUANTUM and have_solution:
    if cycle[1] != quantum_route[1]:
        cycle.reverse()

''' Reverse Brute Force route if opposite to quantum route '''
if COMPUTE_QUANTUM and have_solution:
    if BF_route[1] != quantum_route[1]:
        BF_route.reverse()

''' If no quantum solution reverse Christofides route if opposite to Brute Force route '''
if COMPUTE_QUANTUM or not have_solution:
    if cycle[1] != BF_route[1]:
        cycle.reverse()

if COMPUTE_QUANTUM:
    print("The quantum route of the traveller is :")
    for i in quantum_route[:-1]:
        print(city_names[i], " -> ", end = '')
    print(city_names[0])

print("The brute-force route of the traveller is :")
for i in BF_route[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0])

print("The Simulated Annealing route of the traveller is : ")
for i in cycle[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0])

# Draw the routes
nx.draw_networkx_nodes(G, pos, nodelist = [0], node_size = 100, node_color = "gold", label = mapping)

edge_list_SA = list(nx.utils.pairwise(cycle))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_SA, edge_color = "mediumblue", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.75)

SA_patch = mpatches.Patch(color="mediumblue", label="Simulated Annealing route")

edge_list_BF          = list(nx.utils.pairwise(BF_route))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_BF, edge_color = "green", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.75)

BF_patch = mpatches.Patch(color="green", label="Brute-Force route")

if COMPUTE_QUANTUM:
    edge_list_quantum = list(nx.utils.pairwise(quantum_route))
    nx.draw_networkx_edges(G, pos, edgelist = edge_list_quantum, edge_color = "red" , arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.75)
    quantum_patch = mpatches.Patch(color="red", label="Quantum route")
    ax.legend(handles=[SA_patch, BF_patch, quantum_patch])
else:
    ax.legend(handles=[SA_patch, BF_patch])

plt.title(G.name)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

normalized_spectrum_SA = pd.DataFrame((spectrum_SA["score SA"] - spec_min) / (spec_max - spec_min))
# spectrum_BF = normalized_spectrum_brute_force.plot.hist(ax = ax, column = "score BF", color='slategrey', grid = True, bins = nb_bin)
# spectrum_SA = normalized_df_SA_score.plot.hist(ax = ax, column = "score SA", color='mediumblue', grid = True, bins = nb_bin)

spectrum_Qantum = pd.DataFrame(Q_correct_solution, columns = ["score Quantum", "solution"])
normalized_spectrum_Qantum = pd.DataFrame((spectrum_Qantum["score Quantum"] - spec_min) / (spec_max - spec_min))

concat = pd.concat([normalized_spectrum_brute_force["score BF"], normalized_spectrum_SA["score SA"], normalized_spectrum_Qantum["score Quantum"]], axis=1)
concat.plot.hist(ax = ax, column = ["score BF", "score SA", "score Quantum"], color=['slategrey', 'mediumblue', "red"], grid = True, bins = nb_bin, alpha = 0.5, density = True)
# concat.plot.hist(ax = ax, column = ["score BF", "score SA", "score Quantum"], color=['slategrey', 'mediumblue', "red"], grid = True, bins = nb_bin, alpha = 0.5)

ax.set_title("Normalized score distributions")
ax.set_xlabel("Cycle scores", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Counts"      , labelpad=20, weight='bold', size=12)

score_BF = trunc((best_score_BF - spec_min) / (spec_max - spec_min), decs = round_digit)
ax.axvspan(score_BF, score_BF + spectrum_res, color='green'     , alpha=0.4)
# score_SA = trunc((best_score_SA - spec_min) / (spec_max - spec_min), decs = round_digit)
# ax.axvspan(score_SA, score_SA + spectrum_res, color='mediumblue', alpha=0.5)

SA_patch = mpatches.Patch(color="mediumblue", alpha=0.6, label="Simulated Annealing scores")
BF_patch = mpatches.Patch(color="green"     , alpha=0.4, label="Brute-Force best score")
Q_patch  = mpatches.Patch(color="red"       , alpha=0.4, label="Quantum Annealing scores")
ax.legend(handles=[SA_patch, BF_patch, Q_patch])

plt.tight_layout()
fig.savefig(out_file_BF_spec)
plt.show()

'''********************'''

spectrum_SA                = pd.DataFrame (cycles_SA, columns = ["cycle","score SA"])
normalized_spectrum_SA     = pd.DataFrame((spectrum_SA["score SA"] - spec_min) / (spec_max - spec_min))

spectrum_Qantum            = pd.DataFrame(Q_correct_solution, columns = ["score Quantum", "solution"])
normalized_spectrum_Qantum = pd.DataFrame((spectrum_Qantum["score Quantum"] - spec_min) / (spec_max - spec_min))

# spectrum_BF = normalized_spectrum_brute_force.plot.hist(ax = ax, column = "score BF", color='slategrey', grid = True, bins = nb_bin)
# spectrum_SA = normalized_df_SA_score.plot.hist(ax = ax, column = "score SA", color='mediumblue', grid = True, bins = nb_bin)
concat                     = pd.concat([normalized_spectrum_brute_force["score BF"], normalized_spectrum_SA["score SA"], normalized_spectrum_Qantum["score Quantum"]], axis=1)

plt.title(G.name)
plt.tight_layout()
plt.show()

'''********************'''

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

# ax.hist(normalized_spectrum_brute_force["score BF"], bins = nb_bin, alpha = 0.5, range=[0., 1.0], density = True, histtype='bar', color='slategrey' , label="Brute-Force best score")
# ax.hist(normalized_spectrum_SA["score SA"]         , bins = nb_bin, alpha = 0.5, range=[0., 1.0], density = True, histtype='bar', color='mediumblue', label="Simulated Annealing scores")
# ax.hist(normalized_spectrum_Qantum["score Quantum"], bins = nb_bin, alpha = 0.5, range=[0., 1.0], density = True, histtype='bar', color='red'       , label="Quantum Annealing scores")

# Make a multiple-histogram of data-sets with different length.
x_multi = [normalized_spectrum_brute_force["score BF"], normalized_spectrum_SA["score SA"], normalized_spectrum_Qantum["score Quantum"]]
# ax.hist(x_multi, bins = 100, alpha = 0.5, range=[0., 1.0], density = True)
ax.hist(x_multi, bins = nb_bin , alpha = 0.5, density=True, stacked=False, color=['slategrey','mediumblue','red'], label=["Brute-Force best score","Simulated Annealing scores","Quantum Annealing scores"])

ax.set_title("Normalized score distributions")
ax.set_xlabel("Cycle scores", labelpad=20, weight='bold', size=12)
ax.set_ylabel("Counts"      , labelpad=20, weight='bold', size=12)

score_BF = trunc((best_score_BF - spec_min) / (spec_max - spec_min), decs = round_digit)
ax.axvspan(0., (1./nb_bin), color='green', alpha=0.2)
# score_SA = trunc((best_score_SA - spec_min) / (spec_max - spec_min), decs = round_digit)
# ax.axvspan(score_SA, score_SA + spectrum_res, color='mediumblue', alpha=0.5)

plt.legend(loc='upper right')
plt.tight_layout()
fig.savefig(out_file_BF_spec)
plt.show()
