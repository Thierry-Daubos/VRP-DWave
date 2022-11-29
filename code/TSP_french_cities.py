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
import math
import random
import copy

import matplotlib.pyplot as plt
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
        return int(i*n - i*(i+1)/2 + j - (i+1))

def build_constraint_matrix(n):
    """
     The constraint matrix encodes the constraint that each city (node) is connected to exactly two other cities in the output cycle        
    """
    m = int(n*(n-1)/2)
    C = np.zeros((m,m))
    for i in range(0, n):
        # diagonal terms of C (these are equal to -6)
        for j in range(0, n):
            if i == j:
                continue
            k = index(i, j, n)
            C[k,k] += -5
        # off diagonal terms (these have a bizzare pattern)
        for a in range(0, n):
            for b in range(0, n):
                if a == b or a == i or b == i:
                    continue
                ia = index(i,a,n)
                ib = index(i,b,n)
                C[ia,ib] += 1
    return C

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
    n, _ = M.shape
    # m is the total of binary variables we have to solve for
    # basically given any two nodes, we need to decide if there is an edge connecting them (a binary variable)
    k = 0
    decode = dict()
    for i in range(0, n):
        for j in range(i+1, n):
            decode[k] = [city_names[i], city_names[j]]
            k += 1
    return decode

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

def score(M, X):
    return np.sum(np.multiply(M, X))


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

# Nb_cities = 5
# random.seed(50)

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

scaler = MinMaxScaler(feature_range=(0, 10))
df     = scaler.fit_transform(df)
# df       = df.to_numpy()

# in_file  = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/data/problem1.txt"
out_file = "D:/Documents/Scalian/Quantum_Computing_2022/VRP-DWave/results/solution_quantum-5_cities.txt"

num_samples = 1000

'''
*** Define objective matrix of travel costs ***
# M is the matrix of pairwise costs (i.e. cost to travel from node i to node j)
# M need not be a symmetric matrix but the diagonal entries are ignored and assumed to be zero
'''
M                   = df
Q                   = build_objective_matrix(M)
Decode              = build_decode(M, city_names)
lagrange_multiplier = np.max(np.abs(M))
# lagrange_multiplier = 4000

''' 
*** Encode connectivity constraint in the final cycle ***
# We need to add the constraint that each city is connected to exactly 2 other cities
# We do this using the method of lagrange multipliers where the constraint is absorbed into the objective function
# (this is the hardest part of the problem)
'''
n, _ = M.shape
C    = build_constraint_matrix(n)

'''
*** Define the Hamiltonian of the problem ***
'''
qubo = Q + lagrange_multiplier * C

'''
*** Chose a DWave sampler ***
# QPU sampler to run in production
'''
sampler = EmbeddingComposite(DWaveSampler()) 

'''
*** Run the problem on the QPU recording execution times
'''
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

have_solution  = False
problem_id     = sampleset.info['problem_id']
chain_strength = sampleset.info['embedding_context']['chain_strength']

with open(out_file, 'w') as f:
    f.write(f"Problem Id: {problem_id}\n") # does not depend on sample  
    count = 0
    for e in sampleset.data(sorted_by='energy', sample_dict_cast = False):
        sample               = e.sample
        energy               = e.energy
        num_occurrences      = e.num_occurrences
        chain_break_fraction = e.chain_break_fraction
        X                    = build_solution(sample)
        
        if is_valid_solution(X):
            have_solution = True
            score         = score(M, X)
            f.write(f"Solution:\n")
            f.write(f"{X}\n")
            f.write(f"Score: {score}\n")
            f.write(f"{sample}\n")
            f.write(f"index: {count}\n")
            f.write(f"energy: {energy}\n")
            f.write(f"num_occurrences: {num_occurrences}\n")            
            f.write(f"chain break fraction: {chain_break_fraction}\n")            
            break   # break out of for loop
        count += 1
    f.write(f"chain strength: {chain_strength}\n")  # does not depend on sample
    f.write(f"lagrange multiplier: {lagrange_multiplier}\n")
    f.write(f"Time: {t1-t0:0.4f} s\n")
    
    if not have_solution:
        # https://docs.ocean.dwavesys.com/en/latest/examples/inspector_graph_partitioning.html
        # this is the overall chain break fraction
        chain_break_fraction = np.sum(sampleset.record.chain_break_fraction)/num_samples
        f.write("did not find any solution\n")
        f.write(f"chain break fraction: {chain_break_fraction}\n")

''' decode quantum sample result into route '''
edge_list = list()
for i, city in enumerate(sample):
    if sample[i] :
        edge_list.append(Decode[i])
        
route = list()
route.append(city_names.index(edge_list[0][0]))
route.append(city_names.index(edge_list[0][1]))
firt_city = edge_list[0][0]
next_city = edge_list[0][1]
edge_list.pop(0)

while len(edge_list) > 1:
    for i, edge in enumerate(edge_list):
        if next_city in edge:
            if next_city != edge_list[i][0]:
                route.append(city_names.index(edge_list[i][0]))
                next_city = edge_list[i][0]
            else:
                route.append(city_names.index(edge_list[i][1]))
                next_city = edge_list[i][1]
            edge_list.pop(i)
route.append(city_names.index(firt_city))

print("The quantum route of the traveller is :")
for i in route[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0])

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

fig = plt.gcf()
fig.set_size_inches(30, 30)
# fig.savefig('test2png.png', dpi=100)

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
m.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', service='World_Imagery', verbose=True)


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

# The NetworkX part: put map projection coordinates in pos dictionary
# G = nx.Graph()

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
        G.add_edge(i, j, weight=int(df[i][j]))

pos = {}
for i, name in enumerate(city_names):
    pos[i] = (mx[i], my[i])

# nodes
nx.draw_networkx_nodes(G, pos, node_size = 50, node_color = '#DC143C', label = mapping)

# edges
nx.draw_networkx_edges(G, pos, edge_color = "blue", width = 2.0, alpha = 0.5)

# node labels
pos_higher = {}
y_off      = round(np.mean(my) * 0.02)
for k, v in pos.items():
    pos_higher[k] = (v[0], v[1] + y_off)
    
nx.draw_networkx_labels(G, pos_higher, labels = mapping, font_size = 20, font_family = "sans-serif", font_color = 'crimson')

# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_family = "sans-serif", alpha = 0.5)

# Find best TSP tour
cycle     = nx_app.christofides(G, weight="weight")

''' Reverse Christofides route if opposite to quantum route '''
if cycle[1] != route[1]:
    cycle.reverse()
    
print("The Christofides' route of the traveller is : ")
for i in cycle[:-1]:
    print(city_names[i], " -> ", end = '')
print(city_names[0])


edge_list_chritofides = list(nx.utils.pairwise(cycle))
edge_list_quantum     = list(nx.utils.pairwise(route))

# Draw closest edges on each node only
# nx.draw_networkx_edges(G, pos, edge_color = "red", width = 0.5)

# Draw the route
nx.draw_networkx_nodes(G, pos, nodelist = [0], node_size = 100, node_color = "gold", label = mapping)

nx.draw_networkx_edges(G, pos, edgelist = edge_list_chritofides, edge_color = "gold"        , arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 3.0, alpha = 0.5)
nx.draw_networkx_edges(G, pos, edgelist = edge_list_quantum    , edge_color = "red", arrows = True, arrowstyle = '-|>', arrowsize = 20, width = 3.0, alpha = 0.5)

plt.title(G.name)
plt.tight_layout()
plt.show()


      
# # ax = plt.gca()
# # ax.margins(0.08)
# # plt.axis("off")
# # plt.tight_layout()
# # plt.show()
