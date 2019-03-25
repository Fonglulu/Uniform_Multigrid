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
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            newu[i,j] = (4* u[i,j] -  u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i, j+1])
            
    return newu



def Astencil(u,h):
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            newu[i,j] = (6*u[i,j] +u[i,j+1]+u[i,j-1] + u[i-1,j] + u[i+1, j] +u[i+1,j+1] + u[i-1,j-1])*(h**2)/float(12)
            
    return newu


def G1stencil(u,h):
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            # My version
            newu[i,j] = (-2*u[i,j-1]+2*u[i,j+1] - u[i+1, j] + u[i-1, j] +u[i+1, j+1] - u[i-1, j-1])*h/float(6)
            #newu[i,j] = (u[i,j-1] - u[i,j+1] +2 *u[i+1,j] -2*u[i-1,j] +u[i+1, j+1] -u[i-1,j-1])*h/float(6)
            
            # Linda's version
            #newu[i,j] = (2*u[i,j-1]-2*u[i,j+1] - u[i+1, j] + u[i-1, j] -u[i+1, j+1] + u[i-1, j-1])*h/float(6)
            
            
    return newu



def G2stencil(u,h):
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            # My version
            newu[i,j] = (u[i,j-1] - u[i,j+1] +2 *u[i+1,j] -2*u[i-1,j] +u[i+1, j+1] -u[i-1,j-1])*h/float(6)
            #newu[i,j] = (-2*u[i,j-1]+2*u[i,j+1] - u[i+1, j] + u[i-1, j] +u[i+1, j+1] - u[i-1, j-1])*h/float(6)
            
            # Linda's version
            #newu[i,j] = (u[i,j-1] - u[i,j+1] -2 *u[i+1,j] +2*u[i-1,j] -u[i+1, j+1] +u[i-1,j-1])*h/float(6)
            
            
            
    return newu
            
    
    
    
    
    
    
    



def Rich(u, rhs,alpha):
    
    """ Block Jacobi Method. On each level of grid (same size as initial grid), invoke corresponding matrices
    
    A, L, G1, G2, d and boundaries h1, h2, h3, h4 
    
    """
    
    #print u[0], 'pre'
    
    #print alpha, 'alpha'
    
    #np.set_printoptions(precision=2)
    
    # Get the current size of RHS function 
    [xdim,ydim] = rhs[0][1:-1, 1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
   
    newu  = np.zeros((4, u.shape[1], u.shape[2]))  
    


    #newu  = zeros([u.shape[0], u.shape[0]])
    
    w = 0.2
    
    #print np.shape(rhs[0]), np.shape(u[0])
    newu[0] = u[0] + w*(rhs[0] - np.sqrt(alpha)*Lstencil(u[0]) + G1stencil(u[1], h) + G2stencil(u[2],h))
    
    newu[1] = u[1] + w*(rhs[1] - Lstencil(u[1]) - G1stencil(u[3],h))
    
    newu[2] = u[2] + w*(rhs[2] - Lstencil(u[2]) - G2stencil(u[3],h))
    
    newu[3] = u[3] + w*(rhs[3] - np.sqrt(alpha)*Lstencil(u[3]) - Astencil(u[0],h))
#    
#    u[0] = u[0] + w*(rhs[0] - Lstencil(u[0]) + G1stencil(u[1], h) + G2stencil(u[2],h))
#    
#    u[1] = u[1] + w*(rhs[1] - alpha* Lstencil(u[1]) - G1stencil(u[3],h))
#    
#    u[2] = u[2] + w*(rhs[2] - alpha* Lstencil(u[2]) - G2stencil(u[3],h))
#    
#    u[3] = u[3] + w*(rhs[3] - Lstencil(u[3]) - Astencil(u[0],h))
    
    #print u[0], 'post'
    

    
    
    return newu


    

    
    
    