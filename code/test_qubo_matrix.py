# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:47:30 2023

@author: thierry.daubos
"""
import os
import pandas as pd
import numpy as np
import random
import copy
import time
import minorminer
import networkx as nx

from dwave_qbsolv import QBSolv
from dwave.system import LeapHybridSampler
from dwave.system import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.embedding.chain_strength import scaled
from sklearn.preprocessing import MinMaxScaler

'''
# Read date file of cities
'''
dir_name = "D:\Documents\Scalian\Quantum_Computing_2022\Datasets"
base_filename = "fr"
filename_suffix = "csv"

filename = os.path.join(dir_name, base_filename + '.' + filename_suffix)
df = pd.read_csv(filename)
city_names = list(df.columns.map(str))

global Nb_cities
Nb_cities      = 7

# Function to compute index in Q for variable x_(a,b)
def x(a, b):
    return (a)*Nb_cities+(b)

# Seed value used for ramdomly selecting the cities
Seed_city      = 0
# Scaling factor for the pairwaise cost matrix between nodes
scaling_factor = 1.0
# Number of measurements repetition (up to 10000)
num_samples    = 100
# Sets annealing duration per sample in microseconds (up to 2000)
annealing_time = 100 
# Number of standard deviations used to compute chain strength
N_sigma        = 3.0
# Relative chain strength
RCS            = 1.00
# Relative Lagrange Multiplier
RLM            = 1.00

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

''' Define model's meta-parameters '''
df_km          = copy.deepcopy(df)
scaler         = MinMaxScaler(feature_range=(0., scaling_factor))
df             = scaler.fit_transform(df)

''' Define NetworkX graph '''
G_embedding      = nx.complete_graph(n_nodes)
G_embedding.name = "Graph of " + str(n_nodes) + " french cities"

# Calculating the distances between the nodes as edge's weight.
for i in city_index:
    for j in range(i+1, n_nodes):
        G_embedding.add_edge(i, j, weight = df[i][j])

print("G_embedding ", G_embedding.number_of_nodes(), "Nodes: \n", list(G_embedding.nodes))
print("G_embedding ", G_embedding.number_of_edges(), "Edges: \n", list(G_embedding.edges))
for node in G_embedding.nodes:
    print(node,"->", G_embedding.nodes[node])
for edge in G_embedding.edges:
    print(edge,"->", G_embedding.edges[edge])

max_chain_strength  = np.max(np.abs(G_embedding))
mean_chain_strength = np.mean(np.abs(G_embedding))
std_chain_strength  = np.std(np.abs(G_embedding))

base_chain_strength = mean_chain_strength + (N_sigma * std_chain_strength)
chain_strength      = RCS * base_chain_strength
base_lagrange       = G_embedding.size(weight = 'weight') * G_embedding.number_of_nodes() / G_embedding.number_of_edges()
lagrange_multiplier = RLM * base_lagrange

Q = {}
for i in range(Nb_cities * Nb_cities):
    for j in range(Nb_cities * Nb_cities):
        Q[(i,j)] = 0

print("Q matrix with", len(Q), "entries created.")

# Row Constraints
# The next block sets the constraint that each row has exactly one 1 in our permutation matrix
count_1 = 0
for v in range(Nb_cities):
    for j in range(Nb_cities):
        Q[(x(v,j), x(v,j))] += -1 * lagrange_multiplier
        for k in range(j+1, Nb_cities):
            Q[(x(v,j), x(v,k))] += 2 * lagrange_multiplier
            count_1 +=1

print("Added", Nb_cities,"row constraints to Q matrix.")
print("count_1 :", count_1)

# Column Constraints
# The next block sets the constraint that each column has exactly one 1 in our permutation matrix
count_2 = 0
for j in range(Nb_cities):
    for v in range(Nb_cities):
        Q[(x(v,j), x(v,j))] += -1 * lagrange_multiplier
        for w in range(v+1,Nb_cities):
            Q[(x(v,j), x(w,j))] += 2 * lagrange_multiplier
            count_2 +=1
            
print("Added", Nb_cities,"column constraints to Q matrix.")
print("count_2 :", count_2)

# Objective Function
# Our objective is to minimize the distanced travelled
# The distance we travel from city u to city v in stops 2 and 3 is D(u,v)x_{u,2}x_{v,3}
# This adds D(u,v) to our total distance if we visit city u in stop 2 and city v in stop 3, and adds 0 to our total distance otherwise.
# So, for every pair of cities u and v, we add \sum_{j=1}^{48} D(u,v)x_{u,j}x_{v,j+1} to add the distance travelled from u to v (directly) in our route
# We need to add this for every choice of u and v (and in both directions)

count_3 = 0
for u in range(Nb_cities):
    for v in range(Nb_cities):
        if u != v:
            for j in range(Nb_cities):
                Q[(x(u,j), x(v,(j+1)%Nb_cities))] += df[u][v]
                count_3 +=1

print("Objective function added.")
print("count_3 :", count_3)

start    = time.time()

# sampler   = LeapHybridSampler(solver={'category': 'hybrid'})
# sampleset = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time)


''' !!! WORKING FIXED EMBEDDING !!! '''
random_seed          = 0
max_no_improvement   = 10
tries                = 10
chainlength_patience = 10

qpu             = DWaveSampler(solver={'topology__type': 'pegasus'})
fixed_embedding = minorminer.find_embedding(Q.keys(), qpu.edgelist, random_seed = random_seed, max_no_improvement = max_no_improvement, tries = tries, chainlength_patience = chainlength_patience, verbose = 2)
sampleset       = FixedEmbeddingComposite(qpu, fixed_embedding).sample_qubo(Q, num_reads = num_samples)

df_sampleset = sampleset.to_pandas_dataframe()

''' *************************** '''

end      = time.time()
time_CPU = end - start
print("FixedEmbeddingComposite sampling complete using", time_CPU,"seconds.")

print("sampleset :", sampleset)

# Understanding the results
# Once we run QBSolv, we need to collect and report back the best answer found
# Here we list off the lowest energy solution found and the total distance required for this route
# If you wish to see the cities in order of the route, uncomment out the code at the end of the next cell block.

# First solution is the lowest energy solution found
# sample = next(iter(sampleset))
# Display energy for best solution found
# print('Energy: ', next(iter(sampleset.data())).energy)

for n, sample in enumerate(sampleset):
    valid = True
    route = [-1]*Nb_cities

    for node in sample:
        if sample[node]>0:
            j = node%Nb_cities
            v = (node-j)/Nb_cities
            route[j] = int(v)
    # print(sample)
    # print("route",n," :", route)

    route = [-1]*Nb_cities
    if valid:
        for node in sample:
            if sample[node] > 0:
                j = node % Nb_cities
                v = (node-j) / Nb_cities
                # print("node :", node, "-> index :", j, " - value :", v)
                if route[j] != -1:
                    # print('Stop '+ str(j) +' used more than once.\n')
                    valid = False
                    break
                route[j] = int(v)

    if valid:
        if sum(route) != Nb_cities * (Nb_cities-1)/2:
            # print('Route invalid.\n')
            valid = False
            
    if valid:
        for i in range(Nb_cities):
            if route[i]==-1:
                # print('Stop '+ str(i) +' has no city assigned.')
                valid = False
                break
    if valid:
        print("valid route",n," :", route)
    
        if Seed_city is not None and route[0] != Seed_city and Seed_city in route:
            # rotate to put the starting city in front
            idx   = route.index(Seed_city)
            route = route[idx:] + route[:idx]
            
        print("valid route",n," :", route)
        print("Valid quantum route score is : ")
        for i in route:
            print(city_names[i], " -> ", end = '')
        print(city_names[0])
        
        # Compute and display total mileage
        distance = 0
        for i in range(Nb_cities):
            distance += df_km[route[i]][route[(i+1) % Nb_cities]]
        print('Distance: ', distance, "km")

