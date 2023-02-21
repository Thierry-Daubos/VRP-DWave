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

import dwave.inspector
import dwave_networkx as dnx
from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding.chain_strength import scaled
from dwave.system import FixedEmbeddingComposite
from dwave.system.composites import LazyFixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system import VirtualGraphComposite
from dwave.system import DWaveCliqueSampler

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
COMPUTE_QUANTUM = False

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

# Lagrange multiplier for taking into account the constraint matrix
# lagrange_multiplier = np.max(np.abs(df))
# lagrange_multiplier = 5.0
lagrange_multiplier = 3.038346244970316

# Number of measurements repetition (up to 10000)
num_samples    = 100

# Sets annealing duration per sample in microseconds (up to 2000)
annealing_time = 100 

# Number of standard deviations used to compute chain strength
N_sigma        = 3.0

# Relative chain strength
# chain_strength = RCS * max_chain_strength # 'conservative' value 
# or
# chain_strength = RCS * int(mean_chain_strength + (N_sigma * std_chain_strength)) # 'tighter' value
# The (relative) chain strength to use in the embedding. 
# By default a chain strength of `1.5 sqrt(N)` where `N` is the size of the largest clique, as returned by attribute `.largest_clique_size`
RCS            = 1.0

random.seed(Seed_city)
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
Q                   = build_objective_matrix(M)
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
Q                  <-> HA  : Objective matrix Q = M.X (X mapping Xij to Bk)
C                  <-> HB  : Constraints matrix mapped to Bk
lagrange_mutiplier <-> (B/A)
qubo               <-> H
'''
qubo = Q + lagrange_multiplier * C

print("Cost matrix M        : \n", M, "\n")
print("Objective matrix Q   : \n", Q, "\n")
print("Lagrange multiplier  : \n", lagrange_multiplier, "\n")
print("Constraints matrix C : \n", C, "\n")
print("QUBO matrix qubo     : \n", qubo, "\n")

''' Define NetworkX graph '''
G_embedding      = nx.complete_graph(n_nodes)
G_embedding.name = "Graph of " + str(n_nodes) + " french cities"

# Calculating the distances between the nodes as edge's weight.
for i in city_index:
    for j in range(i+1, n_nodes):
        G_embedding.add_edge(i, j, weight=int(df[i][j]))

print("G_embedding ", len(list(G_embedding.nodes)), "Nodes: \n", list(G_embedding.nodes))
print("G_embedding ", len(list(G_embedding.edges)), "Edges: \n", list(G_embedding.edges))

# Check embedding using the "exact solver" on Ocean API
path                  = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/"
name_Kerberos_map     = "solution_Kerberos_map_cities_"      + str(Nb_cities) + "_seed_" + str(Seed_city)
ext3                  = ".png"
out_file_Kerberos_map = path + name_Kerberos_map + ext3

''' Compute problem's solution by brute-force algorithm '''

path                 = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/"
name_BF_all_pkl      = "solution_BF_all-cities_"      + str(Nb_cities) + "_seed_" + str(Seed_city)
ext                  = ".pkl"
in_file_BF_all_pkl   = path + name_BF_all_pkl + ext
spectrum_brute_force = pd.read_pickle(in_file_BF_all_pkl)

spec_min = spectrum_brute_force["score BF"].min()
spec_max = spectrum_brute_force["score BF"].max()
normalized_spectrum_brute_force = pd.DataFrame((spectrum_brute_force["score BF"] - spec_min) / (spec_max - spec_min))

spectrum_res = 0.005
round_digit  = len(str(int(1./spectrum_res)))-1
spectrum_min = trunc(normalized_spectrum_brute_force.min(), decs = round_digit)
spectrum_max = trunc(normalized_spectrum_brute_force.max(), decs = round_digit)
nb_bin       = int(np.ceil((spectrum_max - spectrum_min)) / spectrum_res)

# solution = unique_solutions[0]
solution      = (spectrum_brute_force["solution"].iloc[0])[0]
BF_route      = decode_route(solution) 
best_score_BF = (spectrum_brute_force["score BF"].iloc[0])

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

# best_SA_cycle = ['Paris', 'Lille', 'Reims', 'Grenoble', 'Nimes', 'Nantes', 'Brest', 'Aix-en-Provence', 'Nice', 'Rennes', 'Strasbourg']
# best_SA_cycle = [0, 10, 2, 6, 4, 1, 5, 7, 8, 9, 3, 0]
# best_SA_cycle = [0, 10, 2, 6, 4, 1, 5, 7, 8, 9, 3, 0]
# best_SA_cycle = [0, 4, 2, 1, 3, 0]
best_SA_cycle = [0, 3, 2, 6, 7, 4, 1, 8, 5,0]

for ind, record in enumerate(spectrum_brute_force.index):
    if (spectrum_brute_force.iloc[ind, 3] == best_SA_cycle):
        print("Brute score :", spectrum_brute_force.iloc[ind, 0])
        normalized_spectrum_Kerberos = (spectrum_brute_force.iloc[ind, 0] - spec_min) / (spec_max - spec_min)
        print("Normalized score :", normalized_spectrum_Kerberos)
        break
    if (spectrum_brute_force.iloc[ind, 3] == list(reversed(best_SA_cycle))):
        best_SA_cycle = spectrum_brute_force.iloc[ind, 3]
        print("Brute score :", spectrum_brute_force.iloc[ind, 0])
        normalized_spectrum_Kerberos = (spectrum_brute_force.iloc[ind, 0] - spec_min) / (spec_max - spec_min)
        print("Normalized score :", normalized_spectrum_Kerberos)        
        break
        
# Draw the routes
nx.draw_networkx_nodes(G, pos, nodelist = [0], node_size = 100, node_color = "gold", label = mapping)

edge_list_SA = list(nx.utils.pairwise(best_SA_cycle))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_SA, edge_color = "mediumblue", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.5)

SA_patch = mpatches.Patch(color="mediumblue", label="QBM Kerberos route")

edge_list_BF          = list(nx.utils.pairwise(BF_route))
nx.draw_networkx_edges(G, pos, edgelist = edge_list_BF, edge_color = "green", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 4.0, alpha = 0.5)

BF_patch = mpatches.Patch(color="green", label="Brute-Force route")

ax.legend(handles=[SA_patch, BF_patch])

plt.title(G.name)
plt.tight_layout()
plt.show()    

fig.savefig(out_file_Kerberos_map)

bool_list = spectrum_brute_force["route"].apply(lambda x: x == best_SA_cycle)
res = [i for i, val in enumerate(bool_list) if val]
print(spectrum_brute_force.iloc[res])

np.where(spectrum_brute_force["route"].apply(lambda x: x == best_SA_cycle))

for ind, record in enumerate(spectrum_brute_force.index):
    # print(spectrum_brute_force.iloc[ind, 3])
    # print('ind :', ind, " route:", list(route), " best_SA_cycle :", best_SA_cycle, "check : ", list(route) == best_SA_cycle)
    # print('ind :', ind, " route:", list(reversed(route)), " best_SA_cycle :", best_SA_cycle, "check : ", list(reversed(route)) == best_SA_cycle)

    if (spectrum_brute_force.iloc[ind, 3] == best_SA_cycle) or (spectrum_brute_force.iloc[ind, 3] == list(reversed(best_SA_cycle))):
        print(spectrum_brute_force.iloc[ind, 0])
       
spectrum_brute_force.index[spectrum_brute_force[:, "route"] == best_SA_cycle].tolist()