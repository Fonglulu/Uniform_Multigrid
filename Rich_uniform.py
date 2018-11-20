#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:18:14 2018

@author: shilu
"""



import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy.sparse.linalg import spsolve
from scipy import eye, zeros, linalg
from numpy import linalg as LA
from copy import copy





def Lstencil(u):
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            u[i,j] = (4* u[i,j] -  u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i, j+1])
            
    return u



def Astencil(u,h):
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            u[i,j] = (6*u[i,j] +u[i,j+1]+u[i,j-1] + u[i-1,j] + u[i+1, j] +u[i+1,j+1] + u[i-1,j-1])*(h**2)/float(12)
            
    return u


def G1stencil(u,h):
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            u[i,j] = (2*u[i,j-1]+2*u[i,j+1] - u[i+1, j] + u[i-1, j] -u[i+1, j+1] + u[i-1, j-1])*h/float(6)
            
            
    return u



def G2stencil(u,h):
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            u[i,j] = (u[i,j-1] - u[i,j+1] -2 *u[i+1,j] +2*u[i-1,j] -u[i+1, j+1] +u[i-1,j-1])*h/float(6)
            
            
    return u
            
            
    
    
    
    
    
    
    



def Rich(u, rhs,alpha):
    
    """ Block Jacobi Method. On each level of grid (same size as initial grid), invoke corresponding matrices
    
    A, L, G1, G2, d and boundaries h1, h2, h3, h4 
    
    """
    
    print alpha, 'alpha'
    
    #np.set_printoptions(precision=2)
    
    # Get the current size of RHS function 
    [xdim,ydim] = rhs[0][1:-1, 1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
    z =  u
    


    
    w = 0.05
    
    u[0] = z[0] + w*(rhs[0] - Lstencil(z[0]) + G1stencil(z[1], h) + G2stencil(z[2],h))
    
    u[1] = z[1] + w*(rhs[1] - alpha* Lstencil(z[1]) + G1stencil(z[3],h))
    
    u[2] = z[2] + w*(rhs[2] - alpha* Lstencil(z[2]) + G2stencil(z[3],h))
    
    u[3] = z[3] + w*(rhs[3] - Lstencil(z[3]) - Astencil(z[0],h))
    
    
    return u


    

    
    
    