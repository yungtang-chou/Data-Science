#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import sys

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    
    
    obj, opt, solution = greedy_networkx(node_count, edges)

    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def greedy_networkx(node_count, edges):
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    G.add_edges_from(edges)
    
    strategies = [nx.coloring.strategy_largest_first,
                  nx.coloring.strategy_random_sequential,
                  nx.coloring.strategy_smallest_last,
                  nx.coloring.strategy_connected_sequential,
                  nx.coloring.strategy_independent_set,
                  nx.coloring.strategy_connected_sequential,
                  nx.coloring.strategy_connected_sequential_dfs,
                  nx.coloring.strategy_connected_sequential_bfs,
                  nx.coloring.strategy_saturation_largest_first]

    best_color_count, best_coloring = node_count, {i: i for i in range(node_count)}
    for strategy in strategies:
        coloring = nx.coloring.greedy_color(G=G, strategy=strategy)
        color_count = max(coloring.values()) + 1
        if color_count < best_color_count:
            best_color_count = color_count
            best_coloring = coloring
    return best_color_count, 0, [best_coloring[i] for i in range(node_count)]
    


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

