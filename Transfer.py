#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:49:58 2019

@author: shilu
"""


# Import appropriate information from other classes
import numpy as np

def Res_Injection(uf):
    
    """
    Restrict the fine grid by simple injection.
    """
    
    # Get the current fine grid
    [depth, xdim, ydim] = uf.shape
    
    uc = uf[:, 0:xdim:2, 0:ydim:2]
    
    return  uc
    
    
    
    
    
    
def FW_Restriction(uf):
    
    """ 
    Restrict fine grid to coarse grid by full weighting operator
    
    Input: current approximation on fine grid uf
    
    Output: Restricted approximation on coarse grid 
    
    """

    # Get the current fine grid
    [depth, xdim, ydim] = uf.shape

    
    # Coarse grid size
    xnodes = int((xdim+1)/2)
    ynodes = int((ydim+1)/2)
    
    # Set coarse grid
    uc = np.zeros((depth, xnodes,ynodes))
    
    
    # Find the values from the original positions
    for k in range(depth):
        for i in range(1, xnodes-1):
            for j in range(1, ynodes-1):
                
                uc[k,i,j] = uf[k,2*i,2*j] + 0.5*(uf[k, 2*i-1, 2*j]+uf[k, 2*i+1, 2*j] + uf[k,2*i, 2*j-1]+\
                    uf[k,2*i,2*j+1]) + 0.25* (uf[k,2*i-1,2*j-1] + uf[k,2*i-1, 2*j+1] + uf[k,2*i+1, 2*j-1] + uf[k,2*i+1, 2*j+1])
                
    return uc
                
              
               

def Interpolation(uc):
    
    """ 
    Interpolate coarse grid to fine grid
    
    Input: current approximation on coarse grid uc

    Output: Interpolated approximation on find grid    
    """
    [depth, xdim, ydim] = uc.shape
    
    # Initialise a next fine grid
    xnodes = 2*xdim-1
    ynodes = 2*ydim-1
    uf = np.zeros((depth, xnodes,ynodes))
    
    
    # For even ordered i and j on fine grid
    for k in range(depth):
        for i in range(xdim):
            for j in range (ydim):
                uf[k, 2*i, 2*j]=uc[k, i,j]
    

    # For even ordered j on fine grid on fine grid
    for k in range(depth):
        for i in range(0, ynodes, 2):
            for j in range(1, xnodes-1, 2):
                uf[k,i,j]=0.5*(uf[k,i,j-1]+uf[k,i,j+1])

        
    # For even ordered i on fine grid on fine grid
    for k in range(depth):
        for i in range(1, xnodes-1, 2):
            for j in range (0, ynodes, 2):
                uf[k,i,j]=0.5*(uf[k,i-1,j]+uf[k,i+1,j])
    
    # For odd ordered i and j on fine grid on fine grid
    for k in range(depth):
        for i in range (1, xnodes-1, 2):
            for j in range (1, ynodes-1, 2):
                uf[k,i,j]=0.25*(uf[k,i-1,j]+uf[k,i+1,j]+uf[k,i,j-1]+uf[k,i,j+1])#    

            
            
    return uf
