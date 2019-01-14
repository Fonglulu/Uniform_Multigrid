#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:17:25 2018

@author: shilu
"""

import numpy as np

from scipy import zeros

from scipy.sparse.linalg import spsolve


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
            
            


def Jacobi(u, rhs,alpha, L):
    
    from copy import copy
    #print u[3], 'pre'
    
    [xdim,ydim] = rhs[0][1:-1, 1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
    crhs = zeros([xdim+2, ydim+2])
    
    g1rhs = zeros([xdim+2, ydim+2])
    
    g2rhs = zeros([xdim+2, ydim+2])
    
    wrhs = zeros([xdim+2, ydim+2])
    
    newu = copy(u)
    

     
    crhs = rhs[0]  + G1stencil(newu[1],h) + G2stencil(newu[2],h)
    
    u[0][1:-1, 1:-1] = np.reshape(spsolve(np.sqrt(alpha)*L, np.reshape(crhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))
    
    g1rhs = rhs[1] - G1stencil(newu[3],h)
    
    u[1][1:-1, 1:-1] = np.reshape(spsolve(L, np.reshape(g1rhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))
    
    g2rhs = rhs[2] - G2stencil(newu[3],h)
    
    u[2][1:-1, 1:-1] = np.reshape(spsolve(L, np.reshape(g2rhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))  
    
    wrhs = rhs[3] - Astencil(newu[0],h)
    
    u[3][1:-1, 1:-1] = np.reshape(spsolve(np.sqrt(alpha)*L, np.reshape(wrhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))  
    
    
    #print u[3], 'post'
    return u
    
    
    
    
            
    
    
    