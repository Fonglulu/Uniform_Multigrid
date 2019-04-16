#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:19:25 2018

@author: shilu
"""
import numpy as np
from functions import Linear
np.set_printoptions(precision=4)



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
    
    """ This rountine assembles dvector from uniformly distributed data
    on uniform grid
    """
    
    
    from scipy import zeros
    
    # initilise the dvector.
    dvector = zeros([(n-2)**2,1])
    
    
    # Ininilise counting .
    IDi = -1
    for i in range(len(list(nodes))):
        
        
        # Find the initerior nodes.
        if nodes[i][0] != 0 and nodes[i][0] != 1 and nodes[i][1] != 0 and nodes[i][1] != 1:
            
            IDi += 1
        
            node1 = nodes[i]
            
            node2 = nodes[i-1]
            
            node3 = nodes[i+n]
            
            node4 = nodes[i+n+1]
            
            node5 = nodes[i+1]
            
            node6 = nodes[i-n]
            
            node7 = nodes[i-n-1]
            
            
            # For each data, locate the triangle
            
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
                
