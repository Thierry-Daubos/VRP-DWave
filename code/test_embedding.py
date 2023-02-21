# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:12:52 2023

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

import dwave.inspector
import dwave_networkx as dnx
from dwave import embedding
from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding.chain_strength import scaled
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
from dwave.system import FixedEmbeddingComposite
from dwave.system.composites import LazyFixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system import VirtualGraphComposite
from dwave.system import DWaveCliqueSampler
from dwave_qbsolv import QBSolv

import dimod
import neal
import minorminer
from minorminer import find_embedding
from minorminer import busclique

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
    Q_linear    = defaultdict(float)
    Q_quadratic = defaultdict(float)

    # Constraint that each row has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            Q_linear[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += 2.0 * lagrange
                Q_linear[((node, pos_1), (node, pos_2))] += 2.0 * lagrange
    # linear coefficients
    # 5 * 25 = 75 lines -> 75
    
    # Constraint that each col has exactly one 1
    for pos in range(N):
        for node_1 in G:
            Q[((node_1, pos), (node_1, pos))] -= lagrange
            Q_quadratic[((node_1, pos), (node_1, pos))] -= lagrange
            for node_2 in set(G)-{node_1}:
                # QUBO coefficient is 2*lagrange, but we are placing this value 
                # above *and* below the diagonal, so we put half in each position.
                Q[((node_1, pos), (node_2, pos))] += lagrange
                Q_quadratic[((node_1, pos), (node_2, pos))] += lagrange
    # quadratic coefficients corresponding to "each column has exactly one 1" constraint
    # 5 * (5*4) = 100 lines -> 175
    
    # Objective that minimizes distance
    for u, v in itertools.combinations(G.nodes, 2):
        for pos in range(N):
            nextpos = (pos + 1) % N

            # going from u -> v
            Q[((u, pos), (v, nextpos))] += G[u][v][weight]
            Q_quadratic[((u, pos), (v, nextpos))] += G[u][v][weight]

            # going from v -> u
            Q[((v, pos), (u, nextpos))] += G[u][v][weight]
            Q_quadratic[((v, pos), (u, nextpos))] += G[u][v][weight]
    # quadratic coefficients corresponding to minimum distance objective
    # 5 * (5*4) = 100 lines -> 275

    return Q, Q_linear, Q_quadratic

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
Nb_cities      = 5

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

''' The objective should be that elements of the qubo matrix are all below the chain strength value '''

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

# Setting chain strength value
# max_chain_strength  = np.ceil(np.max(np.abs(G_embedding)))
# mean_chain_strength = np.ceil(np.mean(np.abs(G_embedding)))
# std_chain_strength  = np.ceil(np.std(np.abs(G_embedding)))

max_chain_strength  = np.max(np.abs(G_embedding))
mean_chain_strength = np.mean(np.abs(G_embedding))
std_chain_strength  = np.std(np.abs(G_embedding))

base_chain_strength = mean_chain_strength + (N_sigma * std_chain_strength)
chain_strength      = RCS * base_chain_strength
base_lagrange       = G_embedding.size(weight = 'weight') * G_embedding.number_of_nodes() / G_embedding.number_of_edges()
lagrange_multiplier = RLM * base_lagrange

# print("Nuber of samples             : ", num_samples)
# print("Annealing time               : ", annealing_time)
# print("Base chain strength          : ", base_chain_strength)
# print("Relative chain strength      : ", RCS)
# print("Targeted chain strength      : ", chain_strength)
# print("Base Lagrange multiplier     : ", base_lagrange)
# print("Relative Lagrange Multiplier : ", RLM)
# print("Lagrange multiplier          : ", lagrange_multiplier)

''' Reference embedding to use: '''
embedding_parameters = {(0, 0): (5154, 5153, 1541),
                        (0, 1): (1721, 1720, 1722),
                        (0, 2): (1482, 5197, 5198, 5199),
                        (0, 3): (1691, 5033, 1692),
                        (0, 4): (1751, 1752, 5018),
                        (1, 0): (5109, 5108, 1556),
                        (1, 1): (5139, 5138),
                        (1, 2): (1647, 4988, 1646, 4989),
                        (1, 3): (1601, 1602, 5048),
                        (1, 4): (1662, 5168),
                        (2, 0): (1736, 1735, 1737),
                        (2, 1): (1781, 1782, 4974),
                        (2, 2): (1616, 4958, 4959, 1617),
                        (2, 3): (5229, 1842, 5228),
                        (2, 4): (5079, 5078),
                        (3, 0): (5063, 1706, 5064),
                        (3, 1): (1827, 1826, 5244),
                        (3, 2): (1676, 1677, 4929),
                        (3, 3): (1586, 5125, 5123, 5124),
                        (3, 4): (5094, 5093),
                        (4, 0): (1797, 5289, 1796, 5169),
                        (4, 1): (1811, 5049, 1812),
                        (4, 2): (1766, 1767, 5003),
                        (4, 3): (5213, 5214, 1857),
                        (4, 4): (5183, 5184, 1632)}


''' Definition of backend to use: '''
qpu     = DWaveSampler(solver={'topology__type': 'pegasus'})
vartype = dimod.Vartype.BINARY

''' Sampler definition stage: '''
base_sampler = dimod.ExactSolver()
sampler      = LazyFixedEmbeddingComposite(qpu, embedding_parameters)

''' QUBO model definition from NetworkX graph: '''
# Not sure about the result of linear and quadratic parts because length don't match!
Q, Q_linear, Q_quadratic = traveling_salesperson_qubo(G_embedding, lagrange = lagrange_multiplier)

print("Q           =", len(Q))
print("Q_linear    =", len(Q_linear))
print("Q_quadratic =", len(Q_quadratic))

''' BQM model definition from QUBO model: '''
bqm_from_qubo_1           = dimod.BinaryQuadraticModel(Q_linear, Q_quadratic, 1.0, vartype)
bqm_from_qubo_1_linear    = bqm_from_qubo_1.get_linear
bqm_from_qubo_1_quadratic = bqm_from_qubo_1.get_quadratic

''' OR'''

bqm_from_qubo_2           = dimod.BinaryQuadraticModel.from_qubo(Q)
bqm_from_qubo_2_linear    = bqm_from_qubo_2.get_linear
bqm_from_qubo_2_quadratic = bqm_from_qubo_2.get_quadratic

''' OR'''

''' BQM model definition from NetworkX graph: '''
G = nx.generators.complete_graph(n_nodes)
for i in range(n_nodes):
    G.add_node(i, bias = 0.0)
for i in city_index:
    for j in range(i+1, n_nodes):
        G.add_edge(i, j, quadratic = df[i][j])

for node in G.nodes:
    print(node,"->", G.nodes[node])
for edge in G.edges:
    print(edge,"->", G.edges[edge])

bqm_from_graph = dimod.from_networkx_graph(G, vartype='BINARY', node_attribute_name='bias', edge_attribute_name='quadratic') # <= working

''' Fixing embedding stage: '''
# structured_sampler   = dimod.StructureComposite(base_sampler, G.nodes, G.edges) <= Ok with base_solver
structured_sampler   = dimod.StructureComposite(sampler, G.nodes, G.edges)

''' Sampling stage: '''

sampleset      = base_sampler.sample(bqm_from_graph) # <= working (but no qubo model is used!)
print("sampleset :\n", sampleset)

''' OR '''

sampleset        = structured_sampler.sample(Q) #<- makes spyder crash!
print("sampleset :\n", sampleset)

# sampleset          = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time)
# sampleset          = LazyFixedEmbeddingComposite.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time)

''' Other approach by reusing a previously computed embedding'''

Q, _, _        = traveling_salesperson_qubo(G_embedding, lagrange = lagrange_multiplier)

# sampler        = LazyFixedEmbeddingComposite(qpu, find_embedding = find_embedding, embedding_parameters = None)
sampler        = EmbeddingComposite(qpu, find_embedding = find_embedding, embedding_parameters = None)
# sampler        = DWaveSampler(qpu, find_embedding = find_embedding, embedding_parameters = None)
# sampler.nodelist
# sampler.edgelist

sampleset      = sampler.sample_qubo(Q, num_reads = num_samples, annealing_time = annealing_time, chain_strength=scaled)
sampleset      = QBSolv().sample_qubo(Q, solver = sampler)

embedding_qubo = (sampleset.info["embedding_context"])["embedding"]
chain_strength = sampleset.info['embedding_context']['chain_strength']

print("embedding_qubo :\n", embedding_qubo)
print("Chain strength :", np.round(chain_strength,4))

# Using LeapHybridSampler as an alternative to obsolete QBSolv
sampler   = LeapHybridSampler(solver={'category': 'hybrid'})
sampleset = sampler.sample_qubo(Q)


embedding_qubo_2 = {(0, 0): (965, 964, 4025, 963),
                    (0, 1): (3845, 935, 934, 3814),
                    (0, 2): (3754, 3755, 889),
                    (0, 3): (3650, 1129, 3649, 1130),
                    (0, 4): (3964, 890, 3965),
                    (1, 0): (1069, 1070),
                    (1, 1): (3710, 949, 3711),
                    (1, 2): (3785, 3784, 3786),
                    (1, 3): (874, 3724, 875, 3725),
                    (1, 4): (904, 905, 3694, 3695),
                    (2, 0): (994, 993, 3890, 950),
                    (2, 1): (3740, 1204, 3739),
                    (2, 2): (3876, 1159, 3815, 1158),
                    (2, 3): (920, 3634, 919, 3635),
                    (2, 4): (1205, 3935, 3934),
                    (3, 0): (3800, 1114, 3799, 1115),
                    (3, 1): (1039, 3860),
                    (3, 2): (1099, 1100),
                    (3, 3): (1084, 1085),
                    (3, 4): (3980, 1040, 3979),
                    (4, 0): (1054, 1055),
                    (4, 1): (3769, 3770, 979),
                    (4, 2): (1009, 1008, 3830),
                    (4, 3): (3905, 3904, 1174),
                    (4, 4): (3920, 3919)}

sampler      = DWaveSampler()

fixed_solver = FixedEmbeddingComposite(sampler, embedding_qubo)
sampleset    = sampler.sample_qubo(Q, solver = fixed_solver, num_reads = num_samples, annealing_time = annealing_time)

embedding_qubo = (sampleset.info["embedding_context"])["embedding"]
chain_strength = sampleset.info['embedding_context']['chain_strength']       

print("embedding_qubo :\n", embedding_qubo)
print("Chain strength              :", np.round(chain_strength,4))

''' Solution from GitHub '''

G         = nx.complete_graph(n_nodes)
sampler   = DWaveSampler()
embed     = find_embedding(G.edges, sampler.edgelist)
print("Embedding found.")

# Now that we have computed our embedding, we pass our sampler (the QPU) and our embedding together into  `FixedEmbeddingComposite`
# We will use the same parameters as in the previous QBSolv `sample_qubo` call.

start = time.time()
resp  = QBSolv().sample_qubo(Q, solver = FixedEmbeddingComposite(sampler, embed), solver_limit=60, timeout=30, num_repeats=1, num_reads = num_samples, chain_strength = chain_strength)
end   = time.time()

time_QPU_Fixed = end-start
print(time_QPU_Fixed, "seconds of wall-clock time.")

''' WORKING FIXED EMBEDDING !!! '''
random_seed          = 0
max_no_improvement   = 10
tries                = 10
chainlength_patience = 10

qpu = DWaveSampler(solver={'topology__type': 'pegasus'})
fixed_embedding = minorminer.find_embedding(Q.keys(), qpu.edgelist, random_seed = random_seed, max_no_improvement = max_no_improvement, tries = tries, chainlength_patience = chainlength_patience, verbose = 2)
sampleset       = FixedEmbeddingComposite(qpu, fixed_embedding).sample_qubo(Q, num_reads = num_samples)