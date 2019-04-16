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
    
    """This routine builds the stencil for L matrix"""
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            newu[i,j] = (4* u[i,j] -  u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i, j+1])
            
    return newu



def Astencil(u,h):
    
    """ This routine builds the stencil for A matrix"""
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            newu[i,j] = (6*u[i,j] +u[i,j+1]+u[i,j-1] + u[i-1,j] + u[i+1, j] +u[i+1,j+1] + u[i-1,j-1])*(h**2)/float(12)
            
    return newu


def G1stencil(u,h):
    
    """This routine builds the stencil for G1 matrix"""
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            # My version
            #newu[i,j] = (-2*u[i,j-1]+2*u[i,j+1] - u[i+1, j] + u[i-1, j] +u[i+1, j+1] - u[i-1, j-1])*h/float(6)
            #newu[i,j] = (u[i,j-1] - u[i,j+1] +2 *u[i+1,j] -2*u[i-1,j] +u[i+1, j+1] -u[i-1,j-1])*h/float(6)
            
            # Linda's version
            newu[i,j] = (2*u[i,j-1]-2*u[i,j+1] - u[i+1, j] + u[i-1, j] -u[i+1, j+1] + u[i-1, j-1])*h/float(6)
            
            
    return newu



def G2stencil(u,h):
    
    """This rountine builds te stencil for G2 matrix"""
    
    newu  = zeros([u.shape[0], u.shape[0]])
    
    for i in range(1, u.shape[0]-1):
        
        for j in range(1, u.shape[0]-1):
            
            # My version
            #newu[i,j] = (u[i,j-1] - u[i,j+1] +2 *u[i+1,j] -2*u[i-1,j] +u[i+1, j+1] -u[i-1,j-1])*h/float(6)
            #newu[i,j] = (-2*u[i,j-1]+2*u[i,j+1] - u[i+1, j] + u[i-1, j] +u[i+1, j+1] - u[i-1, j-1])*h/float(6)
            
            # Linda's version
            newu[i,j] = (u[i,j-1] - u[i,j+1] -2 *u[i+1,j] +2*u[i-1,j] -u[i+1, j+1] +u[i-1,j-1])*h/float(6)
               
            
    return newu
            
    
    
    
    
def TransS2(u,alpha,h):
    
    """ This routine multiplies a given vector u by the transpose of matrix S2.
    the square root alpha verison matrix.
    """
    
    
    newu  = np.zeros((4, u.shape[1], u.shape[2])) 
    
    newu[0] = np.sqrt(alpha)*Lstencil(u[0]) + Astencil(u[3],h)
    
    newu[1] = -G1stencil(u[0],h) + Lstencil(u[1])
    
    newu[2] = -G2stencil(u[0],h) + Lstencil(u[2])
    
    newu[3] = G1stencil(u[1],h) + G2stencil(u[2],h) + np.sqrt(alpha)*Lstencil(u[3])
    
    return newu
    




def MultS2(u,alpha,h):
    
    """ This rountine multiplies a given vector u by the matrix S2, the square 
    root alpha verison matrix.
    
    Input: u: a 4-layer vector 
    """

    newu  = np.zeros((4, u.shape[1], u.shape[2])) 
    
    
    newu[0] = np.sqrt(alpha)*Lstencil(u[0]) - G1stencil(np.sqrt(alpha)*u[1], h) - G2stencil(np.sqrt(alpha)*u[2],h)
    
    newu[1] = Lstencil(np.sqrt(alpha)*u[1]) + G1stencil(np.sqrt(alpha)*u[3],h)
    
    newu[2] = Lstencil(np.sqrt(alpha)*u[2]) + G2stencil(np.sqrt(alpha)*u[3],h)
    
    newu[3] = np.sqrt(alpha)*Lstencil(np.sqrt(alpha)*u[3]) + Astencil(u[0],h)
    
    return newu


def Rich(u, rhs,alpha):
    
    """ Block Jacobi Method. On each level of grid (same size as initial grid), invoke corresponding matrices
    
    A, L, G1, G2, d and boundaries h1, h2, h3, h4 
    """

    
    # Get the current size of RHS function 
    [xdim,ydim] = rhs[0][1:-1, 1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
   
    newu  = np.zeros((4, u.shape[1], u.shape[2]))  
    
    # weighting values depend on alpha
    w = 0.15
    
   
    # Richardson method adapts S3 matrix.
    newu = u + w*(TransS2(rhs, alpha, h)- TransS2(MultS2(u,alpha,h),alpha, h))
    
    # The commented out line implements Richardson method on S2 matrix
    newu = u + w*(rhs - MultS2(u, alpha,h))
    
    return newu


    

    

    
    
    
