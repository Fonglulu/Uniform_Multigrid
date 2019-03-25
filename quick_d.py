#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:50:19 2019

@author: shilu
"""
import time 
import numpy as np
from functions import Linear
np.set_printoptions(precision=4)


i = 6

n= 2**i+1

# Find the spacing
h=1/float(n-1)

# Set the mesh grid, that is interior
#x1, y1 = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))

nodes = np.vstack([x1.ravel(), y1.ravel()]).T



# Set up data
x = np.linspace(0, 1.0,120)
y = np.linspace(0, 1.0,120)
X, Y = np.meshgrid(x,y)

data = Linear(X,Y)
data = data.flatten()

coordx = X.flatten()
coordy = Y.flatten()
Coord = zip(coordx, coordy)


def Polynomial_eval(node1, node2, node3, data_coord):
    
    from math import fabs
    #print data_coord
    x1 = node1[0]
    y1 = node1[1]
    x2 = node2[0]
    y2 = node2[1]
    x3 = node3[0]
    y3 = node3[1]
    
    division = (y1-y2)*(x2-x3)-(y2-y3)*(x1-x2);
    #print division
    
    
    assert fabs(division) > 1.0E-12, "divide by zero"
    
    const = (x3*y2 - y3*x2)/float(division)
    xcoe = (y3-y2)/float(division)
    ycoe = (x2-x3)/float(division)
    
    return data_coord[0]*xcoe+data_coord[1]*ycoe+const


def In_triangle(node1, node2, node3, data_coord):
    
    value1 = Polynomial_eval(node1, node2, node3, data_coord)
    value2 = Polynomial_eval(node2, node1, node3, data_coord)
    value3 = Polynomial_eval(node3, node2, node1, data_coord)
    
    return  (value1 >=0.0 and value2>=0.0 and value3 >=0.0)


def  dvector(Coord, data, nodes,n):
    
    from scipy import zeros
    from copy import copy
    # initilise the dvector.
    dvector = zeros([(n)**2,1])
    
    for i in range(len(Coord)):
        
        data_i = data[i]
        data_coord = Coord[i]
        #print Coord[i]
        
        # difference consists the distances between give data and all grid nodes
        
        difference = Coord[i] - nodes
        
        #print difference, 'difference'
        
        distance = difference[:,0]**2 +difference[:,1]**2
        
        difference = difference.tolist()
        
        distance = distance.tolist()
        
        #print distance, 'distance'
        
        
        node_list = copy(nodes.tolist())
        
        #print node_list, 'node_list'
        
        #print node_list
        
        
        # Find the closest node
        #  the index of third smallest distance 
        node1 = node_list[distance.index(sorted(distance)[0])]
        #print distance.index(sorted(distance)[0]), node1, 'node1'
        
        
        
        
        IDi = node_list.index(node1)
        #print IDi, 'IDi'
        
        # Find the second & third closest node
        
        # Find the index of second smallest distance
        node2 = node_list[distance.index(sorted(distance)[1])]
        
        index2 = distance.index(sorted(distance)[1])
        #print distance.index(sorted(distance)[1]), node2, 'node2'
        
        IDj = node_list.index(node2)
        #print IDj, 'IDj'
        
        
        distance[index2] = distance[index2]+2
        
        
        
        #distance.pop(distance.index(sorted(distance)[1]))
        #print distance
      
        
        # Find the index of third smallest distance
        #node_list_2 = copy(node_list)
        #node_list_2.remove(node2)
        node3 = node_list[distance.index(sorted(distance)[1])]
        #print distance.index(sorted(distance)[1]), node3, 'node3'
        
        IDk = node_list.index(node3)
        #print IDk, 'IDk'
        
        #print node1, node2, node3, Coord[i]
        
        dvector[IDi,0] += Polynomial_eval(node1, node2, node3, data_coord)* data_i
        
        dvector[IDj,0] += Polynomial_eval(node2, node1, node3, data_coord)* data_i
        
        dvector[IDk,0] += Polynomial_eval(node3, node1, node2, data_coord)* data_i
        
        
    return dvector

start = time.time()
dvector =dvector(Coord, data, nodes, n)/float(len(Coord))
done = time.time()
elapsed = done - start
        
        
        
        
        
       
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    