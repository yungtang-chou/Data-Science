#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import networkx as nx
import itertools
import numpy as np
import numba as nb
import time
from random import shuffle
import warnings
from gene import *
warnings.filterwarnings("ignore")

Point = namedtuple("Point", ['x', 'y'])

@nb.jit()
def length(point1, point2):
    l = math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    return(l)

@nb.jit()
def tsplength(solution,points):
    tsp_length = sum([length(points[solution[i - 1]], points[solution[i]]) for i in range(len(points))])
    return(tsp_length)

@nb.jit()
def edgelength(points):
    n = len(points)
    matrix = np.zeros([n,n])
    for i, p in enumerate(points):
        for j, q in enumerate(points): # iterate two points at the same time
            if i == j:
                continue
            elif i < j:
                matrix[i][j] = length(p, q)
            else:
                matrix[i][j] = matrix[j][i] # symmetric matrix 
    return(matrix)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # # build a trivial solution
    # # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)
    # # calculate the length of the tour
    # obj = tsplength(solution,points)
    
    # greedy algorithm
    # solution, obj = greedy(nodeCount, points)

    # dynamic programming
    # solution, obj = dynamic_programming(nodeCount, points)
    
    # k-opt
    # solution, obj = k_opt(nodeCount, points, k_max=2, time_limit=None)
    
    # meta-heuristic restarts
    # solution, obj = meta_heuristic_restarts(nodeCount,points)
    
    # genetic algorithm
    
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(City(float(parts[0]), float(parts[1])))
    popsize = int(len(points) * 100)
    elitesize = int(np.round(len(points) * 0.85,0))
    solution, obj = geneticAlgorithm(population=points, popSize=popsize, eliteSize=elitesize, mutationRate=0.01, generations=500)
    
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

@nb.jit()
def greedy(nodeCount, points):
    g = nx.Graph()
    for i in range(nodeCount):
        for j in range(i+1, nodeCount):
            g.add_edge(i,j, weight = length(points[i],points[j]))
    mst = nx.minimum_spanning_tree(g)
    # From networkx documentation: 
    # Return a minimum spanning tree or forest of an undirected weighted graph.
    # A minimum spanning tree is a subgraph of the graph (a tree) with the minimum sum of edge weights.
    # If the graph is not connected a spanning forest is constructed. A spanning forest is a union of the spanning trees for each connected component of the graph.
    
    solution = list(nx.dfs_preorder_nodes(mst))
    obj = tsplength(solution,points)
    # Produce nodes in a depth-first-search pre-ordering starting from source.
    return(solution, obj)

@nb.jit()
def dynamic_programming(nodeCount, points):
    matrix = edgelength(points)
    solution = held_karp(matrix)[1]
    obj = tsplength(solution,points)
    return(solution, obj)

@nb.jit()
def held_karp(matrix):
    # Implementation of Held-Karp, an algorithm for TSP using dynamic programming.
    n = len(matrix)
    
    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (matrix[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + matrix[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + matrix[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


@nb.jit()
def k_opt(nodeCount, points, k_max=2, time_limit=None):
    solution, obj = greedy(nodeCount, points)
    
    t = time.clock()
    for k in range(2, k_max + 1):
        improved = True
        while improved:
            # if time_limit and time.clock() - t > time_limit:
            #     break
            solution, obj, improved = k_swap_iteration(points, solution, k)
    return(solution, obj)

@nb.jit()
def k_swap_iteration(points, solution, k):
    n_point = len(points)
    solution = solution
    improved = False
    obj = tsplength(solution,points)
    for p in itertools.combinations(range(1, n_point),k):
        new_solution, new_obj = k_swap(solution, obj, p, points)
        if new_obj < obj:
            solution = new_solution
            obj = new_obj
            improved = True
            break
    return(solution, obj, improved)

@nb.jit()
def k_swap(solution, obj, p, points):
    k = len(p) + 1
    segments = [solution[p[i]:p[i + 1]] for i in range(len(p) - 1)]
    best_solution = solution
    best_obj = obj
    for n in range(k):
        for part in itertools.combinations(range(len(segments)),k):
            new_segments = []
            for i, segment in enumerate(segments):
                if i in part:
                    new_segments.append(segment[::-1])
                else:
                    new_segments.append(segment)
            for i, permuted_segments in enumerate(itertools.permutations(new_segments)):
                if n == 0 and i == 0:
                    continue
                new_solution = solution[:p[0]] + \
                    list(itertools.chain.from_iterable(permuted_segments)) + \
                    solution[p[-1] + 1:]
                new_obj = tsplength(new_solution, points)
                if new_obj < best_obj:
                    best_solution = new_solution
                    best_obj = new_obj
    return best_solution, best_obj


def meta_heuristic_restarts(nodeCount, points):
    current_solution = [i for i in range(0, nodeCount)]
    current_obj = tsplength(current_solution,points)
    final_solution = []
    start = []
    i = 0
    while i < 10:
        s = 0
        tmp_solution, tmp_obj = k_opt(nodeCount, points, k_max=2, time_limit=None)
        if tmp_obj < current_obj:
            current_obj = tmp_obj
            final_solution = tmp_solution
        
        shuffle(current_solution)
        if len(start) < int(nodeCount/2):
            while current_solution[0] in start:
                shuffle(current_solution)
            start.append(current_solution[0])
        trivial_obj = tsplength(current_solution, points)
        while trivial_obj > current_obj and s <= int(nodeCount / 2):
            shuffle(current_solution)
            trivial_obj = tsplength(current_solution, points)
            s += 1
        current_obj = trivial_obj
        
        i += 1
    return current_solution, current_obj



import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')