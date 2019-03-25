#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:52:45 2019

@author: shilu
"""

def residue_sqrt_alpha(rhs, u, alpha):
    
    """ This routine takes the approximation  of u, computes the residue by
    r=Au-f, where A is the modified version with square root of alpha.
    """
    
    import numpy as np
    from Rich_uniform import Astencil, G1stencil, G2stencil, Lstencil

    # Get the current size of RHS function
    [xdim,ydim] = rhs[0][1:-1,1:-1].shape
    
    h = 1/ float(xdim+2-1)

    
    # Initialise the residual

    r=np.zeros((4, rhs.shape[1],rhs.shape[2]))
    
    r[0] = rhs[0] - np.sqrt(alpha)*Lstencil(u[0])+ G1stencil(u[1], h) + G2stencil(u[2] ,h)
    
    r[1] = rhs[1] - Lstencil(u[1]) - G1stencil(u[3],h)
    
    r[2] = rhs[2] - Lstencil(u[2]) - G2stencil(u[3], h)
    
    r[3] = rhs[3] - Astencil(u[0], h) - np.sqrt(alpha)* Lstencil(u[3])
    

    
    
    return r


def residue_alpha(rhs,u, alpha):
    
    """ This routine takes the approximation  of u, computes the residue by
    r=Au-f, where A is the original version of matrix.
    """
    
    import numpy as np
    from Rich_uniform import Astencil, G1stencil, G2stencil, Lstencil

    # Get the current size of RHS function
    [xdim,ydim] = rhs[0][1:-1,1:-1].shape
    
    h = 1/ float(xdim+2-1)
    
    r=np.zeros((4, rhs.shape[1],rhs.shape[2]))
#    
    r[0] = rhs[0] - Lstencil(u[0])+ G1stencil(u[1], h) + G2stencil(u[2] ,h)
    
    r[1] = rhs[1] - alpha *  Lstencil(u[1]) - G1stencil(u[3],h)
    
    r[2] = rhs[2] - alpha * Lstencil(u[2]) - G2stencil(u[3], h)
    
    r[3] = rhs[3] - Astencil(u[0], h) -  Lstencil(u[3])
    
    return r

    
