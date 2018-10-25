#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:11:12 2018

@author: shilu
"""

import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy.sparse.linalg import spsolve
from scipy import eye, zeros, linalg
from numpy import linalg as LA
from copy import copy

from MGsolverUp import MGsolverUP

from grid.Grid import Grid

from BuildSquare import build_square_grid
from BuildEquation import build_equation_linear_2D, set_polynomial_linear_2D,\
Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY, NIn_triangle, build_matrix_fem_2D


from grid.function.FunctionStore import zero, exp_soln, exp_rhs, sin_soln, sin_rhs, linear, plain, l2, l3, sin4



def Rich(u, rhs,alpha,A, G1, G2, L):
    
    """ Block Jacobi Method. On each level of grid (same size as initial grid), invoke corresponding matrices
    
    A, L, G1, G2, d and boundaries h1, h2, h3, h4 
    
    """
    
    print alpha, 'alpha'
    
    #np.set_printoptions(precision=2)
    
    # Get the current size of RHS function 
    [xdim,ydim] = rhs[0][1:-1, 1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
    #v = u
    z =  u
    
#    v = np.zeros((4, xdim+2, xdim+2))
#    
#    r = np.zeros((4, xdim+2, xdim+2)) 
#    

    
    w = 0.23

    #h=1/float(xdim+1)

#    c= u[0][1:-1,1:-1]
#    
#
#    g1 = u[1][1:-1,1:-1]
#
#    
#    g2 = u[2][1:-1,1:-1]
#   
#    
#    w = u[3][1:-1,1:-1]
    #print c, 'CCCCCCC'
    
    
    
    
    
    
    
    
#    
    crhs = zeros([xdim+2, ydim+2])
    
    crhs[1:-1, 1:-1] = u[0][1:-1,1:-1]+ w*  (rhs[0][1:-1,1:-1]+ np.reshape(G1 * np.reshape(u[1][1:-1,1:-1], (xdim**2,1)) +G2 * np.reshape(u[2][1:-1,1:-1], (xdim**2,1))-\
                        L *np.reshape(u[0][1:-1,1:-1], (xdim**2,1)), (xdim,xdim) ))
    
    u[0][1:-1,1:-1] = crhs[1:-1,1:-1]
    
    
    
    g1rhs = zeros([xdim+2, ydim+2])
    
    g1rhs[1:-1,1:-1] = u[1][1:-1,1:-1] +w * (rhs[1][1:-1,1:-1]+np.reshape(G1.T*np.reshape(u[3][1:-1,1:-1],(xdim**2,1))-\
                       alpha*L * np.reshape(u[1][1:-1,1:-1], (xdim**2,1)), (xdim,ydim)))
    
    u[1][1:-1,1:-1] =g1rhs[1:-1,1:-1] 
    
    g2rhs = zeros([xdim+2, ydim+2])
    
    g2rhs[1:-1,1:-1] = u[2][1:-1,1:-1] + w* (rhs[2][1:-1,1:-1]+np.reshape(G2.T*np.reshape(u[3][1:-1,1:-1],(xdim**2,1))-\
                   alpha* L * np.reshape(u[2][1:-1,1:-1],(xdim**2,1)) ,(xdim, ydim)))
    
    u[2][1:-1,1:-1] = g2rhs[1:-1,1:-1]
    
    
    wrhs = zeros([xdim+2, ydim+2])
    
    wrhs[1:-1,1:-1] = u[3][1:-1,1:-1] + w* ( rhs[3,1:-1,1:-1]  -np.reshape(A * np.reshape(u[0][1:-1,1:-1], (xdim**2,1))\
                   + L *np.reshape(u[3][1:-1,1:-1], (xdim**2,1)), (xdim, ydim)))
    
    u[3][1:-1,1:-1] = wrhs[1:-1,1:-1]
    
    
# 
###############################################################################  
## Compute Su
###############################################################################  
##    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#
#            v[0,i,j] =    -h * (z[1, i,j-1]  - z[1, i,j+1] -  2* z[1, i-1 ,j] + 2* z[1, i+1,j] - z[1, i-1, j-1] + z[1, i+1, j+1])/float(6)-\
#                                    h *(-2* z[2, i,j-1] + 2 * z[2,i,j+1] + z[2, i-1, j] -  z[2,i+1,j] -z[2,i-1,j-1] + z[2,i+1,j+1])/float(6)-\
#                                    (z[0, i-1,j]+z[0, i+1, j]+z[0, i, j-1]+z[0, i,j+1]- 4*z[0, i,j])
#                                    
#            r[0,i,j]   =    -(rhs[0, i-1,j]+rhs[0, i+1, j]+rhs[0, i, j-1]+rhs[0, i,j+1]- 4*rhs[0, i,j])+\
#                              + h**2 * (6*rhs[3,i,j] +rhs[3,i-1,j]+rhs[3,i,j+1] + rhs[3,i,j-1] + rhs[3,i,j+1]+  rhs[3,i-1,j-1] + rhs[3,i+1,j+1])/float(12*alpha)
#                                    
#                                    
#            
#
#
#    #u[0][1:-1, 1:-1] = (1-omega)* c + omega * np.reshape(spsolve(L, np.reshape(crhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))
#    
#    #u[0][1:-1,1:-1] = crhs[1:-1,1:-1]
#
#    
#    
#    
#    # Get new g1-grid
#
#    #g1rhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#
#                v[1,i,j] =  -h * (-z[3, i,j-1] + z[3, i,j+1] +  2* z[3, i-1 ,j]  -2* z[3, i+1,j] + z[3, i-1, j-1] - z[3, i+1, j+1])/float(6)-\
#                             (z[1, i-1,j]+z[1, i+1, j]+z[1, i, j-1]+z[1, i,j+1]-4*z[1, i,j])
#                             
#                r[1,i,j] = -h * (-rhs[0, i,j-1] + rhs[0, i,j+1] +  2* rhs[0, i-1 ,j]  -2* rhs[0, i+1,j] + rhs[0, i-1, j-1] - rhs[0, i+1, j+1])/float(6)-\
#                              (rhs[1, i-1,j]+rhs[1, i+1, j]+rhs[1, i, j-1]+rhs[1, i,j+1]- 4*rhs[1, i,j])
# 
#
#
#    
#    #u[1][1:-1,1:-1] =  g1rhs[1:-1,1:-1]
#
#
#
#    
#    # Get new g2-grid
#
#    #g2rhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#    
#               v[2,i,j] =  -h * (2* z[3, i,j-1] - 2 * z[3,i,j+1] - z[3, i-1, j] +  z[3,i+1,j] +z[3,i-1,j-1] -z[3,i+1,j+1])/float(6)-\
#                            (z[2, i-1,j]+z[2, i+1, j]+z[2, i, j-1]+z[2, i,j+1]-4*z[2, i,j])
#
#               r[2,i,j] =  -h * (2* rhs[0, i,j-1] - 2 * rhs[0,i,j+1] - rhs[0, i-1, j] +  rhs[0,i+1,j] +rhs[0,i-1,j-1] -rhs[0,i+1,j+1])/float(6)-\
#                         (rhs[2, i-1,j]+rhs[2, i+1, j]+rhs[2, i, j-1]+rhs[2, i,j+1]-4*rhs[2, i,j])
#
#    
#    
#    #u[2][1:-1, 1:-1] =  g2rhs[1:-1,1:-1]
#    
#
#    
#    # Get new w-grid
#
#    #wrhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#    
#            v[3,i,j] = h**2 * (6*z[0,i,j] +z[0,i-1,j]+z[0,i,j+1] + z[0,i,j-1] + z[0,i,j+1]+  z[0,i-1,j-1] + z[0,i+1,j+1])/float(12*alpha)-\
#                        (z[3, i-1,j]+z[3, i+1, j]+z[3, i, j-1]+z[3, i,j+1]-4*z[3, i,j])
#                        
#            r[3,i,j] =  -h * (rhs[1, i,j-1]  - rhs[1, i,j+1] -  2* rhs[1, i-1 ,j] + 2* rhs[1, i+1,j] - rhs[1, i-1, j-1] + rhs[1, i+1, j+1])/float(6)-\
#                                    h *(-2* rhs[2, i,j-1] + 2 * rhs[2,i,j+1] + rhs[2, i-1, j] -  rhs[2,i+1,j] -rhs[2,i-1,j-1] + rhs[2,i+1,j+1])/float(6)
#
#                        
#                        
#                        
#        
# 
###############################################################################
## Sparse solver
###############################################################################                               
#    crhs = zeros([xdim+2, ydim+2])
    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
##    
##
#             #Old Richardson
#            u[0,i,j] =   z[0,i,j] + w* ( rhs[0,i,j] + h * (z[1, i,j-1]  - z[1, i,j+1] -  2* z[1, i-1 ,j] + 2* z[1, i+1,j] - z[1, i-1, j-1] + z[1, i+1, j+1])/float(6)+\
#                                    h *(-2* z[2, i,j-1] + 2 * z[2,i,j+1] + z[2, i-1, j] -  z[2,i+1,j] -z[2,i-1,j-1] + z[2,i+1,j+1])/float(6)+\
#                                    (z[0, i-1,j]+z[0, i+1, j]+z[0, i, j-1]+z[0, i,j+1]- 4*z[0, i,j]))
##            
##            u[0,i,j] =u[0,i,j] + w* (r[0,i,j]+ (v[0, i-1,j]+v[0, i+1, j]+v[0, i, j-1]+v[0, i,j+1]- 4*v[0, i,j])-\
##                               h**2 * (6*v[3,i,j] +v[3,i-1,j]+v[3,i,j+1] + v[3,i,j-1] + v[3,i,j+1]+  v[3,i-1,j-1] + v[3,i+1,j+1])/float(12))
##            
##            
##            
##
##
##    #u[0][1:-1, 1:-1] = (1-omega)* c + omega * np.reshape(spsolve(L, np.reshape(crhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))
##    
##    #u[0][1:-1,1:-1] = crhs[1:-1,1:-1]
##
##    
##    
##    
##    # Get new g1-grid
##
#    #g1rhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#
#                u[1,i,j] =  z[1,i,j] + w*(rhs[1,i,j]+ h * (-z[3, i,j-1] + z[3, i,j+1] + 2* z[3, i-1 ,j] -2* z[3, i+1,j] + z[3, i-1, j-1] - z[3, i+1, j+1])/float(6)+\
#                             (z[1, i-1,j]+z[1, i+1, j]+z[1, i, j-1]+z[1, i,j+1]-4*z[1, i,j]))
#            
##                 u[1,i,j] =u[1,i,j] + w *(r[1,i,j]+ h * (-v[0, i,j-1] + v[0, i,j+1] +  2* v[0, i-1 ,j]  -2* v[0, i+1,j] + v[0, i-1, j-1] - v[0, i+1, j+1])/float(6)+\
##                             (v[1, i-1,j]+v[1, i+1, j]+v[1, i, j-1]+v[1, i,j+1]- 4*v[1, i,j]))
## 
##
##
##    
##    #u[1][1:-1,1:-1] =  g1rhs[1:-1,1:-1]
##
##
##
##    
##    # Get new g2-grid
##
#    #g2rhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
#
#    
#            u[2,i,j] = z[2,i,j] + w*(rhs[2,i,j] +  h * (2* z[3, i,j-1] - 2 * z[3,i,j+1] - z[3, i-1, j] +  z[3,i+1,j] +z[3,i-1,j-1] -z[3,i+1,j+1])/float(6)+\
#                         (z[2, i-1,j]+z[2, i+1, j]+z[2, i, j-1]+z[2, i,j+1]-4*z[2, i,j]))
##            
##            
##             u[2,i,j] = u[2,i,j] + w * (r[2,i,j] +  h * (2* v[0, i,j-1] - 2 * v[0,i,j+1] - v[0, i-1, j] +  v[0,i+1,j] +v[0,i-1,j-1] -v[0,i+1,j+1])/float(6)+\
##                         (v[2, i-1,j]+v[2, i+1, j]+v[2, i, j-1]+v[2, i,j+1]-4*v[2, i,j]))
###            
##            
##            
##            
##
##
##    
##    
##    #u[2][1:-1, 1:-1] =  g2rhs[1:-1,1:-1]
##    
##
##    
##    # Get new w-grid
##
#    #wrhs = zeros([xdim+2, ydim+2])
#    
#    for i in range(1, rhs[0].shape[0]-1):
#        
#        for j in range(1, rhs[0].shape[0]-1):
##
##    
#            u[3,i,j] = z[3,i,j] + w*(rhs[3,i,j] - h**2 * (6*z[0,i,j] +z[0,i-1,j]+z[0,i,j+1] + z[0,i,j-1] + z[0,i,j+1]+  z[0,i-1,j-1] + z[0,i+1,j+1])/float(12*alpha)+\
#                        (z[3, i-1,j]+z[3, i+1, j]+z[3, i, j-1]+z[3, i,j+1]-4*z[3, i,j]))
##
#            u[3,i,j] = u[3,i,j] + w* (r[3,i,j] + h * (v[1, i,j-1]  - v[1, i,j+1] -  2* v[1, i-1 ,j] + 2* v[1, i+1,j] - v[1, i-1, j-1] + v[1, i+1, j+1])/float(6)+\
#                                    h *(-2* v[2, i,j-1] + 2 * v[2,i,j+1] + v[2, i-1, j] -  v[2,i+1,j] -v[2,i-1,j-1] + v[2,i+1,j+1])/float(6))

    #u[3][1:-1, 1:-1] = (1-omega) * w + omega * np.reshape(spsolve(L, np.reshape(wrhs[1:-1,1:-1],(xdim**2, 1))), (xdim, ydim))
    
    #u[3][1:-1, 1:-1] =  wrhs[1:-1,1:-1]
    
    #print u[0], 'Jacobi'
    #print u[3], 'Jacobi'
    
    return u
    
    
    
    