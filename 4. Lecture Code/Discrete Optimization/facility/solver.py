#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
import cvxopt.glpk
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
cvxopt.glpk.options['tm_lim'] = 3600 * 10 ** 3 # one hour

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # # build a trivial solution
    # # pack the facilities one by one until all the customers are served
    # solution = [-1]*len(customers)
    # capacity_remaining = [f.capacity for f in facilities]

    # facility_index = 0
    # for customer in customers:
    #     if capacity_remaining[facility_index] >= customer.demand:
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand
    #     else:
    #         facility_index += 1
    #         assert capacity_remaining[facility_index] >= customer.demand
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand

    # used = [0]*len(facilities)
    # for facility_index in solution:
    #     used[facility_index] = 1
    
    # MIP program
    solution = mip(facilities, customers)
    
    # calculate the cost of the solution
    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    

    # # calculate the cost of the solution
    # obj = sum([f.setup_cost*used[f.index] for f in facilities])
    # for customer in customers:
    #     obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def mip(facilities, customers):
    '''
    ilp(...)
    Solves a mixed integer linear program using GLPK.
    
    (status, x) = ilp(c, G, h, A, b, I, B)
    
    PURPOSE
    Solves the mixed integer linear programming problem
    
        minimize    c'*x
        subject to  G*x <= h
                    A*x = b
                    x[k] is integer for k in I
                    x[k] is binary for k in B
    
    ARGUMENTS
    c            nx1 dense 'd' matrix with n>=1
    
    G            mxn dense or sparse 'd' matrix with m>=1
    
    h            mx1 dense 'd' matrix
    
    A            pxn dense or sparse 'd' matrix with p>=0
    
    b            px1 dense 'd' matrix
    
    I            set of indices of integer variables
    
    B            set of indices of binary variables
    '''
    
    M = len(customers)
    N = len(facilities)
    c = [] # construct the matrix for objective function 
    for j in range(N):
        c.append(facilities[j].setup_cost) # the opening cost regarding whether the stroe is open
    for j in range(N):
        for i in range(M):
            c.append(length(facilities[j].location, customers[i].location)) # distance between facilities and customer
            
    xA = []
    yA = []
    valA = []
    for i in range(M):
        for j in range(N):
            xA.append(i) # For every customer
            yA.append(N + M * j + i) # Each facility 
            valA.append(1) # All the coefficients are 1
    
    b = np.ones(M) # Binary constraint
    
    xG = []
    yG = []
    valG = []
    for i in range(N):
        for j in range(M):
            xG.append(M * i + j) 
            yG.append(i)
            valG.append(-1)
            xG.append(M * i + j)
            yG.append(N + M * i + j)
            valG.append(1)
    
    for i in range(N):
        for j in range(M):
            xG.append(N * M + i)
            yG.append(N + M * i + j)
            valG.append(customers[j].demand)
    
    h = np.hstack([np.zeros(N*M),
                   np.array([fa.capacity for fa in facilities], dtype = 'd')])
    
    binVars = set()
    for var in range(N + M * N):
        binVars.add(var)
        
    status, isol = cvxopt.glpk.ilp(c = cvxopt.matrix(c),
                                   G = cvxopt.spmatrix(valG, xG, yG),
                                   h = cvxopt.matrix(h),
                                   A = cvxopt.spmatrix(valA, xA, yA),
                                   b = cvxopt.matrix(b),
                                   I = binVars,
                                   B = binVars)
    soln = []
    for i in range(M):
        for j in range(N):
            if isol[N + M * j + i] == 1:
                soln.append(j)
    return(soln)

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

