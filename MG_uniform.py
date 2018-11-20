#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:09:26 2018

@author: shilu
"""




import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy.sparse.linalg import spsolve
from scipy import eye, zeros, linalg
from numpy import linalg as LA



from functions import Linear, Xlinear, Ylinear, Zero

from operator import itemgetter
#np.set_printoptions(precision=4)

from copy import copy



import matplotlib.pyplot as plt
from pylab import title, xlabel, ylabel, clf, plot,show, legend
#np.set_printoptions(precision=2)



def Injection(uf):
    
    """ 
    Restrict find grid to coarse grid by injection
    
    Input: current approximation on fine grid uf
    
    Output: Restricted approximation on coarse grid 
    
    """
#    #print uf[1], 'injectbefore'
#    # Get the current size of approximation
    [depth, xdim, ydim] = uf.shape

    
    return uf[:, 0:xdim:2, 0:ydim:2]



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
    
    
    from Rich_uniform import Astencil, G1stencil, G2stencil, Lstencil

    # Get the current size of RHS function
    [xdim,ydim] = rhs[0][1:-1,1:-1].shape
    
    h = 1/ float(xdim+2-1)

    

    
    # Initialise the residual
    #print rhs[0].shape[0], 'RHSDIM'
    r=np.zeros((4, rhs[0].shape[0],rhs[0].shape[1]))
    
    r[0] = rhs[0] - Lstencil(u[0])+ G1stencil(u[1], h) + G2stencil(u[2] ,h)
    
    r[1] = rhs[1] - alpha*Lstencil(u[1]) + G1stencil(u[3],h)
    
    r[2] = rhs[2] - alpha*Lstencil(u[2]) + G2stencil(u[3], h)
    
    r[3] = rhs[3] - Astencil(u[0], h) - Lstencil(u[3])
    
    

                                    
                                  


    return r









def VCycle(u,rhs,  s1, s2, alpha, count = 0):
    
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm
    from Rich_uniform import Rich
    
    # Perfore iterations of smoother to improve the estimate of u.h
    
    #print count
    
    if u[0].shape[0] == 3:
        
        for sweeps in range(s2):
        
             u = Rich(u, rhs,alpha)
             #u =Jacobi(u, rhs, alpha,) 
            
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
             u = Rich(u, rhs, alpha)
             #u = Jacobi(u, rhs, alpha,)
            

#        
            
    if u[0].shape[0] != 3:

        
        rhs = residue(rhs, u, alpha, )
        
        rhs = Injection(rhs)
        

        uc = np.zeros((4, rhs[0].shape[0], rhs[0].shape[0]))
        

        
        uc = VCycle(uc, rhs, s1, s2, alpha, count + 1)
        
        u = u + Interpolation(uc)
            
            
    #print rhs, 'RHS'
    return u
    

    
    
    

def Ultimate_MG(cyclenumber):
    
    from dvector import dvector
    
    from h_boundaries import hboundaries

    


    
    i = 3

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
    datax = np.linspace(0, 1.0,20)
    datay = np.linspace(0, 1.0,20)
  
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

    
    # Set the shape of objective function

    

    # Set penalty term
    alpha = 1


    dvector = dvector(Coord, data, nodes, n)/float(len(Coord))
    
    dvector = np.reshape(dvector, (n-2,n-2))
    
    
    
    h1 = hboundaries(h, n, nodes, intnodes, c, g1, g2, w)[0]
    
    h2 = hboundaries(h, n, nodes, intnodes, c, g1, g2, w)[1]
    
    h3 = hboundaries(h, n, nodes, intnodes, c, g1, g2, w)[2]
    
    h4 = hboundaries(h, n, nodes, intnodes, c, g1, g2, w)[3]
    
    h1 = np.reshape(h1, (n-2,n-2))
    
    h2 = np.reshape(h2, (n-2,n-2))
    
    h3 = np.reshape(h3, (n-2,n-2))
    
    h4 = np.reshape(h4, (n-2,n-2))
    
    


            

    

    
    # Set the initial guess for interior nodes values
    u=np.zeros((4,n,n))

    
    
    
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
        u = VCycle(u,rhs, s1, s2, alpha)
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
        
        
        
        
                
    