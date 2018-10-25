#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 08:17:32 2018

@author: shilu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:57:21 2018

@author: FENG Shi Lu

This module stores the smoothers
"""


import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy.sparse.linalg import spsolve
from scipy import eye, zeros, linalg
from numpy import linalg as LA

from Jacobi import Jacobi
from SOR import SOR
from Rich import Rich
#from JacobiUp import JacobiUp
#from MGsolver import MGsolve
#from MGsolverUp import MGsolverUP

from grid.Grid import Grid
from grid.NodeTable import not_slave_node
from BuildSquare import build_square_grid
from BuildEquation import build_equation_linear_2D, set_polynomial_linear_2D,\
Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY, NIn_triangle, build_matrix_fem_2D
from grid.function.FunctionStore import zero, linear

from operator import itemgetter
#np.set_printoptions(precision=4)

from copy import copy



import matplotlib.pyplot as plt
from pylab import title, xlabel, ylabel, clf, plot,show, legend
#np.set_printoptions(precision=2)


def sin_soln(x,y):
    """Define the function sin(pi x_0)sin(pi x_1)"""
    return sin(2*pi*x)*sin(2*pi*y)

def cos_sin(x, y):
    """Define the function pi cos(pi x_1)sin(pi y_1)"""
    
    return pi*cos(pi*x)*sin(pi*y)

def sin_cos(x, y):
    """Define the function pi sin(pi x_1)cos(pi y_1)"""
    
    return pi*sin(pi*x)*cos(pi*y)

def exp_soln(x,y):
    """Define the function exp(x_0)exp(x_1)"""
    return exp(x)*exp(y)
    

def exp_w(x,y):
    """Define the function -2*exp(x_0)exp(x_1)"""
    return -1.0*exp(x)*exp(y)


def Cos2(x,y):
    
    return cos(2*pi*x)*cos(2*pi*y)

def Xcos2(x,y):
    
    return -2*pi*cos(2*pi*y)*sin(2*pi*x)

def Ycos2(x,y):
    
    return -2*pi*cos(2*pi*x)*sin(2*pi*y)

def XYcos2(x,y):
    
    return (8*0.000000001*pi**2)*cos(2*pi*x)*cos(2*pi*y)

def plain(x,y):
    
    return x-x+1

def Zero(x,y):
    
    return x-x+0.0

def Linear(x,y):
    
    return x+y

def Xlinear(x,y):
    
    return x-x+1.0
    

def Ylinear(x,y):
    
    return y-y+1.0


def L3(x,y):
    
    return x**3 + y**3

def x_3(x,y):
    
    return 3*x**2


def y_3(x,y):
    
    return 3*y**2 

def XYl_3(x,y):
    
     return -0.01*6*(x+y)
 
    

def Exy(x,y):
    return exp(3/((x-0.5)**2+(y-0.5)**2+1))


def Xexy(x,y):
    return -6 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)/((x-0.5)**2+(y-0.5)**2+1)**2
    
def Yexy(x,y):
    return -6 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)/((x-0.5)**2+(y-0.5)**2+1)**2
    
def XYexy(x,y):
    
    return - 1*(36 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)**2/((x-0.5)**2+(y-0.5)**2+1)**4+ \
           24 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)**2/((x-0.5)**2+(y-0.5)**2+1)**3-\
           12 * exp(3/((x-0.5)**2+(y-0.5)**2+1))/((x-0.5)**2+(y-0.5)**2+1)**2+\
           36 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)**2/((x-0.5)**2+(y-0.5)**2+1)**4+\
           24 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)**2/((x-0.5)**2+(y-0.5)**2+1)**3)
           
    



def Injection(uf):
    
    """ 
    Restrict find grid to coarse grid by injection
    
    Input: current approximation on fine grid uf
    
    Output: Restricted approximation on coarse grid 
    
    """
    #print uf[1], 'injectbefore'
    # Get the current size of approximation
    [depth, xdim, ydim] = uf.shape
    
     #Restrict on coarse grid
     
     
    grid = uf
    
    xnodes = int((xdim+1)/2)
    ynodes = int((xdim+1)/2)
    
    for k in range(depth):
        
        grid[k, 0, 0] += 0.5 * uf[k, 0, 1] +0.5 * uf[k,1, 0]
        
        grid[k,0,-1] += 0.5 * uf[k, 0, -2] + 0.5 * uf[k,1,-1]
        
        grid[k,-1,0] += 0.5* uf[k, -1, 1] + 0.5*uf[k,-2,0]
        
        grid[k,-1,-1] += 0.5* uf[k,-2,-1] + 0.5* uf [k, -1,-2]
        
        
        
    for k in range(depth):
        
        for i in range(2, xdim-2, 2):
            
            grid[k, i, 0] = uf[k,i, 0] + 0.5 * uf[k, i-1, 0] +0.5 * uf[k, i+1, 0] + 0.5 * uf[k, i, 1]
            
            grid[k, i, -1] = uf[k, i, -1] + 0.5 * uf[k, i-1, -1] + 0.5 * uf[k, i+1, -1] + 0.5 * uf[k, i, -2]
            
            grid[k, 0, i] = uf[k, 0, i] + 0.5* uf[k, 0, i-1] +0.5 * uf[k, 0, i+1] + 0.5 * uf[k, 1, i]
            
            grid[k, -1, i] =uf[k, -1, i] + 0.5 * uf[k, -1, i-1] + 0.5 * uf[k, -1, i+1] + 0.5 * uf[k, -2, i]
        
    
    for k in range(depth): 
        
        for i in range(2, xdim-2, 2):
            
            for j in range(2, xdim-2, 2):
                
                grid[k, i, j] = 0.25* grid[k, i,j] +0.125* grid[k, i-1,j] +0.125 * grid[k, i+1,j] + 0.125 * grid[k, i,j+1] + 0.125 * grid[k,i, j-1]+\
            0.0625 * grid[k, i-1, j-1] + 0.0625 * grid[k,i-1, j+1] + 0.0625 * grid[k,i+1, j-1] + 0.0625 * grid[k, i+1, j+1]
                
                
    for k in range(depth):
        for i in range(1,xnodes-1):
            for j in range(1,ynodes-1):
                
                grid[k, i,j] = 0.0625 * ( 4* uf[k, 2*i, 2*j]+ 2* uf[k, 2*1 +1, 2*j] + 2* uf[k, 2*i-1, 2*j] + \
                    2 * uf[k, 2*i, 2*j-1] + 2 *uf[k, 2*i, 2*j+1]+  uf[k, 2*i-1, 2*j-1] + uf[k, 2*i-1, 2*j+1]+
                    uf[k, 2*j +1, 2*j-1] + uf[k, 2*i+1, 2*j +1])  
        
    
                
        
        

    
    #grid = copy(uf[:, 0:xdim:2,0:ydim:2])
    

    
    return grid #uf[:, 0:xdim:2, 0:ydim:2]



def Interpolation(uc):
    
    """ 
    Interpolate coarse grid to fine grid
    
    Input: current approximation on coarse grid uc

    Output: Interpolated approximation on find grid    
    """
    [depth, xdim, ydim] = uc.shape
    #print depth, xdim, ydim
    
    # Initialise a next fine grid
    xnodes = 2*xdim-1
    ynodes = 2*ydim-1
    grid = np.zeros((depth, xnodes,ynodes))
    
    
    # For even ordered i and j
    for k in range(depth):
        for i in range(xdim):
            for j in range (ydim):
                grid[k, 2*i, 2*j]=uc[k, i,j]
    

    # For even ordered j  
    for k in range(depth):
        for i in range(0, ynodes, 2):
            for j in range(1, xnodes-1, 2):
                grid[k,i,j]=0.5*(grid[k,i,j-1]+grid[k,i,j+1])

        
    # For even ordered i   
    for k in range(depth):
        for i in range(1, xnodes-1, 2):
            for j in range (0, ynodes, 2):
                grid[k,i,j]=0.5*(grid[k,i-1,j]+grid[k,i+1,j])
    
    # For odd ordered i and j
    for k in range(depth):
        for i in range (1, xnodes-1, 2):
            for j in range (1, ynodes-1, 2):
                grid[k,i,j]=0.25*(grid[k,i-1,j]+grid[k,i+1,j]+grid[k,i,j-1]+grid[k,i,j+1])#    

            
            
    return grid







def residue(rhs, u, alpha):

    # Get the current size of RHS function
    [xdim,ydim] = rhs[0][1:-1,1:-1].shape
    
    h = 1/ float(xdim+2-1)

    
#    c=np.reshape(u[0][1:-1,1:-1], (xdim**2,1))

#    g1 = np.reshape(u[1][1:-1,1:-1], (xdim**2,1))
#    
#    g2 = np.reshape(u[2][1:-1,1:-1], (xdim**2,1))
#
#    w = np.reshape(u[3][1:-1,1:-1], (xdim**2,1))
    
#    crhs = np.reshape(G1 * g1 + G2 * g2, (xdim, ydim))
#    
#    g1rhs = np.reshape( G1.T*  w ,(xdim, ydim))
#    
#    g2rhs = np.reshape( G2.T*  w ,(xdim, ydim))
    
    #wrhs = np.reshape(A * c, (xdim, ydim))
#


    
    # Initialise the residual
    #print rhs[0].shape[0], 'RHSDIM'
    r=np.zeros((4, rhs[0].shape[0],rhs[0].shape[1]))
    
    #print crhs, 'rrr'
    print rhs[3], 'RHS f', u[3], 'aprrox'
    for i in range(1, rhs[0].shape[0]-1):
        for j in range(1, rhs[0].shape[0]-1):
            
            # rhs[0] = f1 +G1g1 + G2g2
#            r[0,i,j] = rhs[0, i,j]+crhs[i-1,j-1] + (u[0, i-1,j]+u[0, i+1, j]+u[0, i, j-1]+u[0, i,j+1]-4*u[0, i,j])
            

#            r[1,i,j] = (rhs[1, i,j] +g1rhs[i-1,j-1]) + alpha * (u[1, i-1,j]+u[1, i+1, j]+u[1, i,j-1]+u[1, i,j+1]-4*u[1, i,j])
#            
#            r[2,i,j] = (rhs[2, i,j] + g2rhs[i-1,j-1]) + alpha * (u[2, i-1,j]+u[2, i+1,j]+u[2, i,j-1]+u[2, i,j+1]-4*u[2,i,j])
#            
#            r[3,i,j] = rhs[3, i,j] -wrhs[i-1,j-1] + (u[3, i-1,j]+u[3, i+1, j]+u[3, i,j-1]+u[3, i,j+1]-4*u[3,i,j])
#            
            # Use stencil of G1 and G2
            r[0,i,j] = rhs[0,i,j] +h * (u[1, i,j-1] - u[1, i,j+1] -  2* u[1, i-1 ,j] + 2* u[1, i+1,j] - u[1, i-1, j-1] + u[1, i+1, j+1])/float(6)+\
                                    h * (-2* u[2, i,j-1] + 2 * u[2,i,j+1] + u[2, i-1, j] -  u[2,i+1,j] -u[2,i-1,j-1] +u[2,i+1,j+1])/float(6)+\
                                  (u[0, i-1,j]+u[0, i+1, j]+u[0, i, j-1]+u[0, i,j+1]-4*u[0, i,j])
                                  
            r[1,i,j] = rhs[1,i,j] + h* (-u[3, i,j-1] + u[3, i,j+1] + 2*u[3, i-1 ,j] -2* u[3, i+1,j] + u[3, i-1, j-1] -u[3, i+1, j+1])/float(6)+\
                                     alpha*(u[1, i-1,j]+u[1, i+1, j]+u[1, i,j-1]+u[1, i,j+1]-4*u[1, i,j])
                                    
            r[2,i,j] = rhs[2,i,j] + h* (2*u[3, i,j-1] - 2*u[3,i,j+1] - u[3, i-1,j] +  u[3,i+1,j] +u[3,i-1,j-1] -u[3,i+1,j+1])/float(6)+\
                                   alpha*(u[2, i-1,j]+u[2, i+1,j]+u[2, i,j-1]+u[2, i,j+1]-4*u[2,i,j])
                                   
                                   
            #r[3,i,j] = rhs[3, i,j] -wrhs[i-1,j-1] + (u[3, i-1,j]+u[3, i+1, j]+u[3, i,j-1]+u[3, i,j+1]-4*u[3,i,j])
            r[3,i,j] = rhs[3,i,j] -h**2 * (6*u[0,i,j] +u[0,i-1,j]+u[0,i,j+1] + u[0,i,j-1] + u[0,i,j+1]+  u[0,i-1,j-1] + u[0,i+1,j+1])/float(12)+\
                                   (u[3, i-1,j]+u[3, i+1, j]+u[3, i,j-1]+u[3, i,j+1]-4*u[3,i,j])
                                    
                                    
                                  

                                
            
    print r[3], 'residue'
    
    
    
    
    #print r[0], 'Residual'
    return r









def VCycle(u,rhs,  s1, s2, alpha, A_list, G1_list, G2_list, L_list, count = 0):
    
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm
    
    # Perfore iterations of smoother to improve the estimate of u.h
    
    #print count
    
    if u[0].shape[0] == 3:
        
        for sweeps in range(s2):
        
             #u = Rich(u, rhs,alpha, A_list[count], G1_list[count], G2_list[count],  L_list[count])
             u =Jacobi(u, rhs, alpha, L_list[count]) 
            
    else:
        
        for sweeps in range(s1):
#            
#            fig = plt.figure(figsize=(8,5))
#            ax = fig.gca(projection='3d')
#            
#            h = 1/float(u[0].shape[0]-1)
#            
#            x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
#        
#            
#        #uexact = sin_soln(x1, y1)
#        # Plot the surface.
#            surf = ax.plot_surface(x1, y1, u[0], cmap=cm.coolwarm,
#                               linewidth=0)
#        
#        # Customize the z axis.
#        #ax.set_zlim(-1.01, 1.01)
#            ax.zaxis.set_major_locator(LinearLocator(10))
#            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#            fig.colorbar(surf, shrink=0.5, aspect=5)
#            print 'here comes plot'
#            plt.show() 
        
            
            #print count, 'count'
             #u = Rich(u, rhs, alpha, A_list[count], G1_list[count], G2_list[count],  L_list[count])
             u = Jacobi(u, rhs, alpha, L_list[count])
            
#            fig = plt.figure(figsize=(8,5))
#            ax = fig.gca(projection='3d')
#            
#            h = 1/float(u[0].shape[0]-1)
#            
#            x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
#        
#            
#        #uexact = sin_soln(x1, y1)
#        # Plot the surface.
#            surf = ax.plot_surface(x1, y1, u[0], cmap='viridis',
#                               linewidth=0, antialiased=False)
#        
#        # Customize the z axis.
#        #ax.set_zlim(-1.01, 1.01)
#            ax.zaxis.set_major_locator(LinearLocator(10))
#            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#            fig.colorbar(surf, shrink=0.5, aspect=5)
#            plt.show()
#        
            
    if u[0].shape[0] != 3:

        
        rhs = residue(rhs, u, alpha, )
        
        rhs = Injection(rhs)
        

        uc = np.zeros((4, rhs[0].shape[0], rhs[0].shape[0]))
        

        
        uc = VCycle(uc, rhs, s1, s2, alpha, A_list, G1_list, G2_list, L_list, count + 1)
        
        u = u + Interpolation(uc)
            
            
    #print rhs, 'RHS'
    return u
    

    
    
    

def Ultimate_MG(cyclenumber):
    
    from TPSFEM import h1, h2, h3, h4, Amatrix, G1, G2, Lmatrix,  dvector
    
    # Set up the grid size， this includes boundary

    grid = Grid()
    
    i = 8

    n= 2**i+1
    
    # Find the spacing
    h=1/float(n-1)
    
    # Set the mesh grid, that is interior
    #x1, y1 = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
    x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
    #print x1, y1
    # Set the data space
    datax = np.linspace(0, 1.0,20)
    datay = np.linspace(0, 1.0,20)
  
    dataX, dataY = np.meshgrid(datax,datay)
    

    # Set the exact solution of c, g1, g2, w on every node
    cexact = Linear
    
    g1exact =  Xlinear
    
    g2exact = Ylinear
    
    wexact = Zero
    #Initial = Sin2
    
    # Set the shape of objective function
    Z = cexact(dataX, dataY)
    
    
    data = Z.flatten()
    

    # Set penalty term
    alpha = 0.0000000001
    L_list = []
    
    A_list = []
    
    G1_list = []
    
    G2_list = []
    
    xdim = []
    
    xdim.append(n)
    
    levelnumber  = 0
    
    
    
    
    while ( (xdim[levelnumber]-1) % 2 == 0 and xdim[levelnumber]-1 >2) :
    
    
            levelnumber = levelnumber+1
            
            xdim.append((xdim[levelnumber - 1] -1) //2 +1)
    
  
      
    for i in xdim :
        
        if i == n:
        
            # Initialise grid for calculating matrices and boundaries
            grid = Grid()
            
            # Build square 
            build_square_grid(i, grid, zero)
            
            
            
            # Store matrices on grid
            build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  dataX, dataY)
            
            print 'Set-up'
    
            
        
            # Find the boundary for Ac + Lw = d - h1
            h1 = np.reshape(h1(grid,cexact, wexact), (n-2,n-2))
            #h1 = h1(grid,cexact, wexact)
            
            # Find the boundary for alphaLg1 -G1^Tw = -h2
            h2 = np.reshape(h2(grid, g1exact, wexact, alpha), (n-2,n-2))

            #h2 = h2(grid, g1exact, wexact, alpha)
            
            # Find the boundary for alphaLg2 -G2^Tw = -h3
            h3 = np.reshape(h3(grid, g2exact, wexact, alpha), (n-2,n-2))
            #h3 = h3(grid, g2exact, wexact, alpha)
            
            # Find the boundary for Lc -G1g1 -G2g2 = -h4
            h4 = np.reshape(h4(grid, cexact, g1exact, g2exact), (n-2,n-2))
            #h4 = h4(grid, cexact, g1exact, g2exact)
            
            # Find d vector
            #print dvector(grid, data)
            
            print '1'
            
            dvector = np.reshape(dvector(grid, data), (n-2,n-2))
            
            print 'd', dvector
            
           
            
            L_list.append(Lmatrix(grid))
            
            
            A_list.append(Amatrix(grid))
            
            
            G1_list.append(G1(grid))
    
            
            G2_list.append(G2(grid))
#            

        
        else :
            
            # Initialise grid for calculating matrices and boundaries
            grid = Grid()
            
            # Build square 
            build_square_grid(i, grid, zero)
            
            # Store matrices on grid
            build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  dataX, dataY)
            

            
            L_list.append(Lmatrix(grid))
            
            print '2'
            
            A_list.append(Amatrix(grid))
            
            
            G1_list.append(G1(grid))
    
            
            G2_list.append(G2(grid))
            
    #print len(A_list)
            

    

    
    # Set the initial guess for interior nodes values
    u=np.zeros((4,n,n))
    #u[0]= Initial(x1,y1)
    #print u, u[0], 'u0'
    
    
    
    # Set RHS at intilisation
    #rhs = np.zeros((4, n-2, n-2))
    rhs = np.zeros((4,n,n))
    rhs[0][1:-1,1:-1] = -h4
    rhs[1][1:-1,1:-1] = -h2
    rhs[2][1:-1,1:-1] = -h3
    rhs[3][1:-1,1:-1]= dvector-h1
    #print rhs[3][1:-1,1:-1]

    
    # Set the boundary (also the exact solution in this case)
#    
    u[0,0,:] = cexact(x1, y1)[0]
    
    u[0, -1,:] = cexact(x1, y1)[-1]
    
    u[0, :, 0] = cexact(x1, y1)[:, 0]
    
    u[0,:,-1] = cexact(x1, y1)[:,-1]
    
    u[1,0,:] = g1exact(x1, y1)[0]
    
    u[1, -1,:] = g1exact(x1, y1)[-1]
    
    u[1, :, 0] = g1exact(x1, y1)[:,0]
    
    u[1,:,-1] = g1exact(x1, y1)[:,-1]
    
    u[2,0,:] = g2exact(x1, y1)[0]
    
    u[2, -1,:] = g2exact(x1, y1)[-1]
    
    u[2, :, 0] = g2exact(x1, y1)[:,0]
    
    u[2,:,-1] = g2exact(x1, y1)[:,-1]
    
    u[3,0,:] = wexact(x1, y1)[0]
    
    u[3, -1,:] = wexact(x1, y1)[-1]
    
    u[3, :, 0] = wexact(x1, y1)[:,0]
    
    u[3,:,-1] = wexact(x1, y1)[:,-1]
    
#    u2 =copy(u)
    # Set the number of relax
    s1=2
    s2=2
#    s3=6
#    s4=6
    
    
#    #Initialise a list to record l2 norm of resudual 
    rnorm=[np.linalg.norm(residue(rhs, u, alpha)[0, 1:-1,1:-1])*h] #A_list[0], G1_list[0], G2_list[0])[0,2:-2,2:-2]) * h]
##    
##    # Initialise a list to record l2 norm of error
#    ecnorm = [np.linalg.norm(u[0]-cexact(x1, y1))*h]
#    eg1norm = [np.linalg.norm(u[1]-g1exact(x1, y1))*h]
##    egg1norm = [np.linalg.norm(u2[1]-g1exact(x1, y1))*h]
##    e3norm = [np.linalg.norm(u[2]-g2exact(x1, y1))*h]
#    ewnorm = [np.linalg.norm(u[3]-wexact(x1, y1))*h]
#    
    
    
    # Start V-cycle
    for cycle in range(1, cyclenumber+1):
        
        #print rhs[0], 'rhs'
        u = VCycle(u,rhs, s1, s2, alpha,A_list, G1_list, G2_list, L_list)
 #       eg1norm.append(np.linalg.norm((u[1]-g1exact(x1,y1))[2:-2,2:-2])*h)
#        u2 = VCycle(u2,rhs, s3, s4, alpha, A_list, G1_list, G2_list, L_list)
    
        
        #print rhs[0], 'RHS'
        rnorm.append(np.linalg.norm(residue(rhs, u, alpha)[0,2:-2,2:-2])*h) #A_list[0], G1_list[0], G2_list[0])[0,2:-2,2:-2])*h)
#    
#        ecnorm.append(np.linalg.norm((u[0]-cexact(x1,y1))[2:-2,2:-2])*h) 
#         eg1norm.append(np.linalg.norm((u[1]-g1exact(x1,y1))[2:-2,2:-2])*h)
##        egg1norm.append(np.linalg.norm((u2[1]-g1exact(x1,y1))[2:-2,2:-2])*h)
#        
#        
#        ewnorm.append(np.linalg.norm(u[3]-wexact(x1,y1))*h)
        


        
     #Plot the semi-log for resiudal and erro

    xline = np.arange(cyclenumber+1)
    plt.figure(figsize=(4,5))
    plt.semilogy(xline, rnorm, 'bo-', xline, rnorm, 'k',label='sdad')
    #plt.semilogy(xline, egg1norm, 'bo', xline, egg1norm, 'k',label='sdad')
    title('Convergence with Error(Jacobi)')
    xlabel('Number of cycles')
    ylabel('Error under l2 norm')
    plt.show()
    print u, rnorm

    return u
    
    

    # Set the RHS 
    
    
    
def Plot_approximation(cyclenumber):   
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')
    
    # Make data.
    i = 5
    
    n= 2**i+1
    
    h=1/float(n-1)
    
    x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
    
    #u = sin(pi*x1)*sin(pi*y1) + 5*sin(31*pi*x1)*sin(31*pi*y1)
    u = Ultimate_MG(cyclenumber)[0]
    print(u)
    
    #uexact = exp_soln(x1, y1)
     #Plot the surface.
#    surf = ax.plot_surface(x1, y1, u, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    
#    # Customize the z axis.
#    #ax.set_zlim(-1.01, 1.01)
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
     
    ax.plot_surface(x1, y1, u,cmap='viridis',linewidth=0)
    
    # Set the z axis limits
    #ax.set_zlim(node_v.min(),node_v.max())
    
    # Make the ticks looks pretty
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.show()    
    
    
# =============================================================================
#  Injection   
# =============================================================================

#    grid = uf
    
#       xnodes = int((xdim+1)/2)
#       ynodes = int((xdim+1)/2)
#    
#    for k in range(depth):
#        
#        grid[k, 0, 0] += 0.5 * uf[k, 0, 1] +0.5 * uf[k,1, 0]
#        
#        grid[k,0,-1] += 0.5 * uf[k, 0, -2] + 0.5 * uf[k,1,-1]
#        
#        grid[k,-1,0] += 0.5* uf[k, -1, 1] + 0.5*uf[k,-2,0]
#        
#        grid[k,-1,-1] += 0.5* uf[k,-2,-1] + 0.5* uf [k, -1,-2]
#        
#        
#        
#    for k in range(depth):
#        
#        for i in range(2, xdim-2, 2):
#            
#            grid[k, i, 0] = uf[k,i, 0] + 0.5 * uf[k, i-1, 0] +0.5 * uf[k, i+1, 0] + 0.5 * uf[k, i, 1]
#            
#            grid[k, i, -1] = uf[k, i, -1] + 0.5 * uf[k, i-1, -1] + 0.5 * uf[k, i+1, -1] + 0.5 * uf[k, i, -2]
#            
#            grid[k, 0, i] = uf[k, 0, i] + 0.5* uf[k, 0, i-1] +0.5 * uf[k, 0, i+1] + 0.5 * uf[k, 1, i]
#            
#            grid[k, -1, i] =uf[k, -1, i] + 0.5 * uf[k, -1, i-1] + 0.5 * uf[k, -1, i+1] + 0.5 * uf[k, -2, i]
#        
#    
#    for k in range(depth): 
#        
#        for i in range(2, xdim-2, 2):
#            
#            for j in range(2, xdim-2, 2):
#                
#                grid[k, i, j] = 0.25* grid[k, i,j] +0.125* grid[k, i-1,j] +0.125 * grid[k, i+1,j] + 0.125 * grid[k, i,j+1] + 0.125 * grid[k,i, j-1]+\
#            0.0625 * grid[k, i-1, j-1] + 0.0625 * grid[k,i-1, j+1] + 0.0625 * grid[k,i+1, j-1] + 0.0625 * grid[k, i+1, j+1]
                
                
#    for k in range(depth):
#        for i in range(1,xnodes-1):
#            for j in range(1,ynodes-1):
#                
#                grid[k, i,j] = 0.0625 * ( 4* uf[k, 2*i, 2*j]+ 2* uf[k, 2*1 +1, 2*j] + 2* uf[k, 2*i-1, 2*j] + \
#                    2 * uf[k, 2*i, 2*j-1] + 2 *uf[k, 2*i, 2*j+1]+  uf[k, 2*i-1, 2*j-1] + uf[k, 2*i-1, 2*j+1]+
#                    uf[k, 2*j +1, 2*j-1] + uf[k, 2*i+1, 2*j +1])  
        
        
        
        
                
    