#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:19:25 2018

@author: shilu
"""
import numpy as np
from functions import Linear

#i = 2
#
#n= 2**i+1
#
## Find the spacing
#h=1/float(n-1)
#
## Set the mesh grid, that is interior
##x1, y1 = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
#x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
#
#nodes = np.vstack([x1.ravel(), y1.ravel()]).T
#
#
#
## Set up data
#x = np.linspace(0, 1.0,20)
#y = np.linspace(0, 1.0,20)
#X, Y = np.meshgrid(x,y)
#
#data = Linear(X,Y)
#data = data.flatten()
#
#coordx = X.flatten()
#coordy = Y.flatten()
#Coord = zip(coordx, coordy)


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
    
    dvector = zeros([(n-2)**2,1])
    
    
    IDi = -1
    for i in range(len(list(nodes))):
        
        
        
        if nodes[i][0] != 1 and nodes[i][0] != 0 and nodes[i][1] != 0 and nodes[i][1] != 1:
            
            IDi += 1
            
            print IDi
    
            
        
            node1 = nodes[i]
            
            node2 = nodes[i-1]
            
            node3 = nodes[i+n]
            
            node4 = nodes[i+n+1]
            
            node5 = nodes[i+1]
            
            node6 = nodes[i-n]
            
            node7 = nodes[i-n-1]
            
            
            
            for j in range(len(Coord)):
                
                if In_triangle(node1, node2, node3, Coord[j]):
                    
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node2, node3, Coord[j])* data[j]
                
                if In_triangle(node1, node3, node4, Coord[j]):
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node3, node4, Coord[j])* data[j]
                    
                if In_triangle(node1, node4, node5, Coord[j]):
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node4, node5, Coord[j])* data[j]
                    
                if In_triangle(node1, node5, node6, Coord[j]):
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node5, node6, Coord[j])* data[j]
                    
                if In_triangle(node6, node1, node7, Coord[j]):
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node6, node7, Coord[j])* data[j]
                    
                if In_triangle(node1, node7, node2, Coord[j]):
                    
                    dvector[IDi,0] += Polynomial_eval(node1, node7, node2, Coord[j])* data[j]
                
                
    return dvector

#dvector =dvector(Coord, data, nodes, n)/float(len(Coord))
                