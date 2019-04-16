#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:09:26 2018

@author: shilu
"""

# Import appropriate information from other classes
import time 

from Transfer import Interpolation, Res_Injection, FW_Restriction
from Residue import residue_sqrt_alpha
from PlotRountines import smoother_plot, convergence_plot
import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy.sparse.linalg import spsolve
from scipy import eye, zeros, linalg
from numpy import linalg as LA


from functions import Linear, Xlinear, Ylinear, Zero, Exy, Xexy, Yexy, XYexy

from operator import itemgetter


from copy import copy



def VCycle(u,rhs,  s1, s2, alpha):
    """ This routine implements the recurssion version of V-cycle
        input: current approximation of u
               rhs 
               s1: number of iterations for relaxtions applied downwards
               s2: number of iterations for relaxtions applied upwards
               alpha
        outpu: new approximation of u
    """
    
    from Rich_uniform import Rich
    
    
    if u[0].shape[0] != 3:
        
        for sweeps in range(s1):
            #print 'downwards'
            u = Rich(u, rhs, alpha)
        
        rhs1 = residue_sqrt_alpha(rhs, u, alpha)

        
        rhs1 = Res_Injection(rhs1)
        
        uc = np.zeros((4, rhs1[0].shape[0], rhs1[0].shape[0]))
        
        uc = VCycle(uc, rhs1, s1, s2, alpha, )
        
        u = u + Interpolation(uc)
        
        
    
    for sweeps in range(s2):
        
             u = Rich(u, rhs, alpha)
        
    return u
    

    
def Setup_Grid(i, alpha, matrix_type):
    
    """ This routine sets up the grid(mesh). The grid is the uniform mesh with 
        uniformly distributed data. Be aware that the number of data need to be 
        entered manually
        
        input: i the index of grid size
               alpha
               matrix_type, if is the original version, enter 'Original'.
               
        There is no return of this routine
    """
    
    from dvector import dvector
    
    from h_boundaries import hboundaries
    
    global h, u, rhs, cexact, x1, y1
    
    n= 2**i+1
    
    # Find the spacing
    h=1/float(n-1)
    
    # Set the mesh grid with data structure of nnumpy array 
    x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
    nodes = np.vstack([x1.ravel(), y1.ravel()]).T
    
    # Set the initerior of the mesh
    intx, inty = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
    intnodes = np.vstack([intx.ravel(), inty.ravel()]).T
    
    # Set the uniformly distributed data 
    datax = np.linspace(0, 1.0,30)
    
    datay = np.linspace(0, 1.0,30)
    
    dataX, dataY = np.meshgrid(datax,datay)

    data = Linear(dataX,dataY)
    
    data = data.flatten()
    
    coordx = dataX.flatten()
    
    coordy = dataY.flatten()
    
    Coord = zip(coordx, coordy)

    # Set the exact solution of c, g1, g2, w on every node
    cexact = Linear
    c = cexact(x1,y1).flatten()
    g1exact =  Xlinear
    g1 = g1exact(x1,y1).flatten()
    g2exact = Ylinear
    g2 = g2exact(x1,y1).flatten()
    wexact = Zero
    w = wexact(x1,y1).flatten()
    
    # Counting the time to build dvector
    start = time.time()
    
    # Import the dvector
    dvector = dvector(Coord, data, nodes, n)/float(len(Coord))
    
    dvector = np.reshape(dvector, (n-2,n-2))
    
    done = time.time()
    
    elapsed = done - start
   
    
    
    # import the boundary condition to form rhs vector
    h1 = hboundaries(alpha, h, n, nodes, intnodes, c, g1, g2, w)[0]
    
    h2 = hboundaries(alpha, h, n, nodes, intnodes, c, g1, g2, w)[1]
    
    h3 = hboundaries(alpha, h, n, nodes, intnodes, c, g1, g2, w)[2]
    
    h4 = hboundaries(alpha, h, n, nodes, intnodes, c, g1, g2, w)[3]
    
    h1 = np.reshape(h1, (n-2,n-2))
    
    h2 = np.reshape(h2, (n-2,n-2))
    
    h3 = np.reshape(h3, (n-2,n-2))
    
    h4 = np.reshape(h4, (n-2,n-2))
    
    u=np.zeros((4,n,n))
    
    if matrix_type =='Original':
    
        rhs = np.zeros((4,n,n))
        rhs[0][1:-1,1:-1] = -h4
        rhs[1][1:-1,1:-1] = -h2
        rhs[2][1:-1,1:-1] = -h3
        rhs[3][1:-1,1:-1]= dvector-h1
        
    else:
        
        rhs = np.zeros((4,n,n))
        rhs[0][1:-1,1:-1] = -np.sqrt(alpha)*h4
        rhs[1][1:-1,1:-1] = -h2/float(np.sqrt(alpha))
        rhs[2][1:-1,1:-1] = -h3/float(np.sqrt(alpha))
        rhs[3][1:-1,1:-1]= dvector-h1
    
    



def Ultimate_MG(cyclenumber, i, alpha, matrix_type):
    """ This routine takes the grid set up by the function Setup_Grid and invoke the
    defined V-cycle to solve the matrix
    
    Input: Cyclenumber: the number of V-cycle
                     i: index of grid size
                     alpha
                     matrix_type
    """



    global u, rhs, h, cexact, x1, y1
    Setup_Grid(i, alpha, matrix_type)



    # Set the number of relaxation
    s1=20
    s2=20

    #Initialise a list to record l2 norm of resudual 
    rnorm=[np.linalg.norm(residue_sqrt_alpha(rhs, u, alpha)[0])*h] 
    enorm = [np.linalg.norm((u[0]-cexact(x1, y1))[1:-1,1:-1])*h]
    
    # Start V-cycle
    for cycle in range(1, cyclenumber+1):

        u = VCycle(u,rhs, s1, s2, alpha)
        # Compute the residual after each cycle
        rnorm.append(np.linalg.norm(residue_sqrt_alpha(rhs, u, alpha)[0])*h) 
        # Compute the error on c after each cycle
        enorm.append(np.linalg.norm((u[0]-cexact(x1,y1))[1:-1,1:-1])*h)
        
    #Plot the semi-log for resiudal and erro r
    convergence_plot(cyclenumber,rnorm)

    

    return u

    
