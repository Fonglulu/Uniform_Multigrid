#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 19:35:25 2018

@author: shilu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 19:44:03 2018

@author: shilu
"""
import timeit
import numpy as np
from numpy import  pi, sin, cos, exp, inf
from scipy import zeros, linalg
from scipy.sparse import csc_matrix, lil_matrix, bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import linalg as LA
import math



from grid.Grid import Grid
from BuildSquare import build_square_grid
from BuildEquation import build_equation_linear_2D, set_polynomial_linear_2D,\
Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY, NIn_triangle, build_matrix_fem_2D

from Triangle import triangle_iterator, Interior_triangle_iterator
from grid.NodeTable import node_iterator, not_slave_node
from grid.ConnectTable import connect_iterator 

from BuildSquare import build_square_grid_matrix
from grid.function.FunctionStore import zero, exp_soln, exp_rhs, sin_soln, sin_rhs, linear, plain, l2, l3, sin4, exy, cos2
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

from PlotPackman import plot_fem_grid
from operator import itemgetter
from copy import copy


alpha = 0.0001

def Linear(x,y):
    
    return x+y

def Plain(x,y):
    
    return x-x+1

def L2(x,y):
    
    return x**2 + y**2

def Zero(x,y):
    
    return x-x


def Xlinear(x,y):
    
    return 3*x**2
    

def Ylinear(x,y):
    
    return 3*y**2

def x_2(x,y):
    
    return 2*x

def y_2(x,y):
    
    return 2*y

def w_l2(x,y):
    
    return -2

def Sin(x,y):
    
    return sin(pi*x)*sin(pi*y)

def Xsin(x,y):
    
    return pi*cos(pi*x)*sin(pi*y)

def Ysin(x,y):
    
    return pi*cos(pi*y)*sin(pi*x)

def XYsin(x,y):
    
    return -(alpha *pi*pi* cos(pi*x)*cos(pi*y))

def L3(x,y):
    
    return x**3 + y**3

def x_3(x,y):
    
    return 3*x**2


def y_3(x,y):
    
    return 3*y**2 

def XYl_3(x,y):
    
     return -6*alpha*(x+y)
 
    
def Exy(x,y):
    return exp(3/((x-0.5)**2+(y-0.5)**2+1))


def Xexy(x,y):
    return -6 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)/((x-0.5)**2+(y-0.5)**2+1)**2
    
def Yexy(x,y):
    return -6 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)/((x-0.5)**2+(y-0.5)**2+1)**2
    
def XYexy(x,y):
    
    return -alpha*(36 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)**2/((x-0.5)**2+(y-0.5)**2+1)**4+ \
           24 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+x)**2/((x-0.5)**2+(y-0.5)**2+1)**3-\
           12 * exp(3/((x-0.5)**2+(y-0.5)**2+1))/((x-0.5)**2+(y-0.5)**2+1)**2+\
           36 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)**2/((x-0.5)**2+(y-0.5)**2+1)**4+\
           24 * exp(3/((x-0.5)**2+(y-0.5)**2+1))*(-0.5+y)**2/((x-0.5)**2+(y-0.5)**2+1)**3)
    
    
def Cos2(x,y):
    
    return cos(2*pi*x)*cos(2*pi*y)

def Xcos2(x,y):
    
    return -2*pi*cos(2*pi*y)*sin(2*pi*x)

def Ycos2(x,y):
    
    return -2*pi*cos(2*pi*x)*sin(2*pi*y)

def XYcos2(x,y):
    
    return (8*alpha*pi**2)*cos(2*pi*x)*cos(2*pi*y)
           

    



bone  = np.loadtxt(r"/Users/shilu/Desktop/SPline/FEM_tut_coding/triangle/bone.txt")

#Create grid and multivariate normal
x = np.linspace(0, 1.0,20)
y = np.linspace(0, 1.0,20)

X, Y = np.meshgrid(x,y)

Z2 = Exy(X,Y)

##Make a 3D plot
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z2,cmap='viridis',linewidth=0)
#ax.set_xlabel('X axis')
#ax.set_ylabel('Y axis')
#ax.set_zlabel('Z axis')
#plt.show()
##

data = Z2.flatten()
coordx = X.flatten()
coordy = Y.flatten()
Coord = zip(coordx, coordy)

#data = bone[:,2]
#coordx = bone[:,0]
#coordy = bone[:,1]
#Coord=zip(coordx, coordy)


##############################################################################
##############################################################################

grid = Grid()
#
## Build a 9*9 grid
i =7
n = 2**i+1

h =1/float(n-1)


#
true_soln = zero
#

# Boundares
Crhs = Exy
#
g1rhs = Xexy
#
g2rhs = Yexy
#
wrhs = XYexy
#
#
#
build_square_grid(n, grid, true_soln)
#
build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)







Nl=[]
for node in (not_slave_node(grid)):
    
    Nl.append([node,node.get_node_id()])
    
Nl=sorted(Nl, key = itemgetter(1))

Nl=[node[0] for node in Nl]
        
for node in Nl:
    node.set_value(Nl.index(node))

        
#



# Generate Amatrix
    
def Amatrix(grid):
    

#        node.set_value(Nl.index(node))
    
#        
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
            
    for node in Nl:
        node.set_value(Nl.index(node))
            
    Amatrix = csr_matrix((len(Nl), len(Nl)))
    
    for node in node_iterator(grid):
        
        if not node.get_slave():
            
            idi = int(node.get_value())
            
            #print node.get_value(), node.get_node_id()._id_no
            
            for endpt in connect_iterator(grid, node.get_node_id()):
                
                idj = int(grid.get_value(endpt))
                
                #print endpt._id_no, grid.get_value(endpt)
                
                aentry = grid.get_matrix_value(node.get_node_id(), endpt)[1]
                
                #print aentry
                
                if not grid.get_slave(endpt):
                    
                    Amatrix[idi, idj] =aentry


    return Amatrix/float(len(Coord))

#
Amatrix = Amatrix(grid).todense()
       
     
            
# Generate h1 boundary condition Ac + Lw = d - h1

def h1(grid, Crhs, wrhs): 
    

    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
            
    for node in Nl:
        node.set_value(Nl.index(node))
        
    
    h1 = zeros((len(Nl), 1))
        
    h = 1/float(math.sqrt(len(Nl))+1)
    
    for node in not_slave_node(grid):
        
        # Ignore slave (or boundary) nodes
    
            # Which row corresponds to the current node?
            i = int(node.get_value())
            
            #print node.get_value(), node.get_node_id()._id_no
            
            
            for endpt in connect_iterator(grid, node.get_node_id()):
                
                #j = int(grid.get_value(endpt))
                
                #print j
                
            
                if grid.get_slave(endpt):
                    
                    
                    coord = grid.get_coord(endpt)
            
                    c = Crhs(coord[0], coord[1])
                    
                    w = wrhs(coord[0], coord[1])

                    

                    
#                    if (grid.get_value(endpt) - node.get_value()) == 0:
#                        
#                        h1[i] += c * h**2* 6/float(12) + w * 4
#                        
#                        
#                    elif abs(grid.get_value(endpt) -node.get_value()) ==1:
#                        
#                        h1[i] += c * h ** 2 /float(12) + w * (-1)
#                        
#                    elif abs(grid.get_value(endpt) -node.get_value()) == int(math.sqrt(len(Nl))):
#                        
#                        h1[i] += c * h **2 /float(12) + w * (-1)
#                        
#                    elif abs(grid.get_value(endpt) -node.get_value()) == int(math.sqrt(len(Nl))) + 1:
#                        
#                        h1[i] += c * h**2 /float(12)
                    
                    
                    
                    #print aentry
                    
                    lentry = grid.get_matrix_value(node.get_node_id(), endpt)[0]
            
                    aentry = grid.get_matrix_value(node.get_node_id(), endpt)[1]/float(len(Coord))
            
                    h1[i] += c* aentry + w * lentry
    return h1

                
h1 =h1(grid, Crhs, wrhs)


# Generate h2 boundary condition alphaLg1 -G1^Tw = -h2
    
def h2(grid, g1rhs, wrhs, alpha):
    
#    g1rhs = zero
#    
#    wrhs =zero
    
#    build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
            
    for node in Nl:
        node.set_value(Nl.index(node))
    
    
    h2= zeros([len(Nl), 1])
    
    for node in not_slave_node(grid):
        
        # Ignore slave (or boundary) nodes
    
            # Which row corresponds to the current node?
            i = int(node.get_value())
            
            for endpt in connect_iterator(grid, node.get_node_id()):
                
            
                if grid.get_slave(endpt):
                    
                    coord = grid.get_coord(endpt)
                    
                    #print coord
            
                    g1 = g1rhs(coord[0], coord[1])
                    #print g1
                    
                    w = wrhs(coord[0], coord[1])
                    
                    print  #w
                    
                    #print endpt._id_no
            
                    lentry = grid.get_matrix_value(node.get_node_id(), endpt)[0]
                    
                    g1entry = grid.get_matrix_value(node.get_node_id(), endpt)[2]
                    
                    #print aentry
            
                    h2[i] += alpha * g1 * lentry - w * g1entry #G1, G2
                    #h2[i] += alpha * g1 * lentry +w * g1entry # -G1, -G2
                    
    return h2


h2 = h2(grid, g1rhs, wrhs, alpha)


# Generate h3 boundary condition alphaLg2 -G2^Tw = -h3

def h3(grid, g2rhs, wrhs, alpha):
    
#    g2rhs = zero
#    
#    wrhs = zero
#    
#    build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
#    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
 
    for node in Nl:
        node.set_value(Nl.index(node))
    
    
    h3= zeros([len(Nl), 1])
    
    for node in not_slave_node(grid):
        
        # Ignore slave (or boundary) nodes
    
            # Which row corresponds to the current node?
            i = int(node.get_value())
            
            for endpt in connect_iterator(grid, node.get_node_id()):
                
            
                if grid.get_slave(endpt):
                    
                    coord = grid.get_coord(endpt)
            
                    g2 = g2rhs(coord[0], coord[1])
                    
                    w = wrhs(coord[0], coord[1])
                    
                    #print c, w
                    
                    #print endpt._id_no
            
                    lentry = grid.get_matrix_value(node.get_node_id(), endpt)[0]
                    
                    # G2 entries on boundary
                    g2entry = grid.get_matrix_value(node.get_node_id(), endpt)[3]
                    
                    #print aentry
                    
                    h3[i] += alpha * g2* lentry - w* g2entry # G1, G2
                    #h3[i] += alpha * g2* lentry + w* g2entry  # -G1, -G2
                    
    return h3

h3 = h3(grid, g2rhs, wrhs, alpha)

# Generate h4 boundary condition for Lc -G1g1 -G2g2 = -h4 on boundary
    
def h4(grid, Crhs, g1rhs, g2rhs):
    
#    Crhs = plain
#    
#    g1rhs = zero
#    
#    g2rhs = zero
#
#    build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    for node in Nl:
        node.set_value(Nl.index(node))
    
    
    h4 = zeros([len(Nl),1])
    
    for node in not_slave_node(grid):
    
    
            # Which row corresponds to the current node?
            i = int(node.get_value())
            
            for endpt in connect_iterator(grid, node.get_node_id()):
                
            
                if grid.get_slave(endpt):
                    
                    coord = grid.get_coord(endpt)
                    
            
                    c = Crhs(coord[0], coord[1])
                    
                    g1 = g1rhs(coord[0], coord[1])
                    
                    g2 = g2rhs(coord[0], coord[1])
                    
                    #print c
                    
                    #print endpt._id_no
            
                    lentry = grid.get_matrix_value(node.get_node_id(), endpt)[0]
                    
                    g1entry = grid.get_matrix_value(node.get_node_id(), endpt)[2]
                    
                    g2entry = grid.get_matrix_value(node.get_node_id(), endpt)[3]
                    
                    #print aentry
                    
                    
            
                    h4[i] += c* lentry + g1entry * g1  + g2entry * g2 # G1, G2
                    #h4[i] += c* lentry - g1entry * g1  - g2entry * g2 # -G1, -G2
                    
                    #print h4
                    
    return h4

h4 = h4(grid, Crhs, g1rhs, g2rhs)


def Lmatrix(grid):
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    for node in Nl:
        node.set_value(Nl.index(node))
#    
    
    
    Lmatrix = csr_matrix((len(Nl), len(Nl)))
    
    for node in node_iterator(grid):
    
        # Ignore slave (or boundary) nodes
        if not node.get_slave():
            
            # Which row corresponds to the current node?
            i = int(node.get_value())
        
            for endpt1 in connect_iterator(grid, node.get_node_id()):
    
                    # Which column corresponds to the current node?
                    j = int(grid.get_value(endpt1))
                    
                    # What is the corresponding matrix value (in the FEM grid)
                    lentry = grid.get_matrix_value(node.get_node_id(), endpt1)[0] 
        
                    # We must not include slave nodes in the matrix columns
                    if not grid.get_slave(endpt1):
                        Lmatrix[i, j] = lentry
    return Lmatrix
    

Lmatrix = Lmatrix(grid).todense()              

                
# Generate G1 matrix
def G1(grid):
    
    #build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    for node in Nl:
        node.set_value(Nl.index(node))
    
    
    
    G1 = csr_matrix((len(Nl), len(Nl)))
    
    for node in node_iterator(grid):
    
        # Ignore slave (or boundary) nodes
        if not node.get_slave():
            
            # Which row corresponds to the current node?
            i = int(node.get_value())
        
            for endpt1 in connect_iterator(grid, node.get_node_id()):
    
                    # Which column corresponds to the current node?
                    j = int(grid.get_value(endpt1))
                    
                    # What is the corresponding matrix value (in the FEM grid)
                    g1 = grid.get_matrix_value(node.get_node_id(), endpt1)[2] 
        
                    # We must not include slave nodes in the matrix columns
                    if not grid.get_slave(endpt1):
                        G1[i, j] = g1
    return G1

G1 = G1(grid).todense()

# Generate G2 matrix
    
def G2(grid):
    
    
    #build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    for node in Nl:
        
        node.set_value(Nl.index(node))

    G2 = csr_matrix((len(Nl), len(Nl)))
    
    for node in node_iterator(grid):
    
        # Ignore slave (or boundary) nodes
        if not node.get_slave():
            
            # Which row corresponds to the current node?
            i = int(node.get_value())
        
            for endpt1 in connect_iterator(grid, node.get_node_id()):
    
                    # Which column corresponds to the current node?
                    j = int(grid.get_value(endpt1))
                    
                    # What is the corresponding matrix value (in the FEM grid)
                    g2 = grid.get_matrix_value(node.get_node_id(), endpt1)[3] 
        
                    # We must not include slave nodes in the matrix columns
                    if not grid.get_slave(endpt1):
                        G2[i, j] = g2
                    
                    
    return G2



G2 = G2(grid).todense()
#print LA.cond(G2.todense())

                
                
# Generate dvector   

def dvector(grid, data):
    
    
    #build_matrix_fem_2D(grid, Poisson_tri_integrate, TPS_tri_intergrateX, TPS_tri_intergrateY,  X, Y)
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    for node in Nl:
        node.set_value(Nl.index(node))
  


               
    dvector = zeros([len(Nl),1])
    
    for tri in Interior_triangle_iterator(grid):
        
        if tri[1].get_node_id() < tri[2].get_node_id():
            
            basis1 = set_polynomial_linear_2D(tri[0], tri[2], tri[1])
            
            Idi = int(tri[0].get_value())
            
            
            for i in range(len(Coord)):
                
                
                if NIn_triangle(tri[0] , tri[1], tri[2], Coord[i]):
                    
                        
                        
                        dvector[Idi,0] += basis1.eval(Coord[i][0], Coord[i][1]) * data[i]
                        
                        
                    
    return dvector/float(len(Coord))
#
dvector = dvector(grid, data)


dvector1 = dvector - h1
##
#
#
def Lets_Make_the_Damn_Big_Matrix():
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
    
    ZeroMatrix = csr_matrix((len(Nl),len(Nl)))
    
    BigMat = bmat([[Amatrix, ZeroMatrix, ZeroMatrix, Lmatrix],\
                       [ZeroMatrix, alpha*Lmatrix, ZeroMatrix, G1.T],\
                       [ZeroMatrix, ZeroMatrix, alpha*Lmatrix,  G2.T],\
                       [Lmatrix, G1, G2, ZeroMatrix]])
    return BigMat
    
BigMat = Lets_Make_the_Damn_Big_Matrix()
#print LA.cond(BigMat.todense())
#
#w,v=LA.eig(BigMat.todense().T* BigMat.todense())
#print 'max', max(w), 'min' ,min(w)
#
#wmin = w.argmax()
#vmin = v[:,wmin]
#vmin=vmin.real
#vmin=vmin[0:len(Nl)]
#x_a = range(len(Nl))
#plt.plot(x_a,vmin)

one = zeros([4*len(Nl),1])
one[0:len(Nl)]=dvector1
one[len(Nl):2*len(Nl)]=-h2
one[2*len(Nl):3*len(Nl)]=-h3
one[3*len(Nl):4*len(Nl)]=-h4

value_vector =spsolve(BigMat, one)
error_vector = copy(value_vector)[0:len(Nl)]





for node in node_iterator(grid):
    
    # The error is only calculated at the interior nodes
    
    if not node.get_slave():
        
        # What row in the matrix does the current node correspond to
        i = int(node.get_value())
        

        value = value_vector[i]
        coord = node.get_coord()
        error_vector[i] = error_vector[i]-l3(coord)
        #print value
        
        #print value

        
        # Record the value at the current node
        node.set_value(value)
        
        
        
    # If the node is a slave node, its value should be given by the boundary condition
    else:
        node.set_value(exy(node.get_coord()))
#        
#
#print "Error norm :", linalg.norm(error_vector)*h
#    
  

def plot_fem_solution(grid):
    from grid.Grid import Grid
    from grid.NodeTable import node_iterator
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.pyplot import show, figure
    from scipy.interpolate import griddata
    from numpy import mgrid, array
#   

   
# The connection values are set separately with respect to the location, 

    #grid=Grid()
#    
    #Find the position of the nodes and the values
    node_x=[]
    node_y=[]
    node_v=[]
    for node in node_iterator(grid):
        coord = node.get_coord()
        node_x.append(coord[0])
        node_y.append(coord[1])
        node_v.append(node.get_value())
        
    # Store the results in an array
    
    node_x = array(node_x)
    node_y = array(node_y)
    node_v = array(node_v)
    
    #print('node_x',node_x)
    #print('node_value', node_v)
    
    
    # Initialise the figure
    fig = plt.figure(figsize=(8,5)) 
    ax = fig.gca(projection='3d') 
    
    
    # Interpolate the nodes onto a structured mesh
#    X, Y = mgrid[node_x.min():node_x.max():10j,
#                 node_y.min():node_y.max():10j]
    
    X, Y = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))
    
    Z = griddata((node_x,node_y), node_v, (X,Y), method='cubic')
    
    
    # Make a surface plot
    ax.plot_surface(X, Y, Z,cmap='viridis',
                       linewidth=0)
#    surf=ax.plot_surface(X, Y, Z,cmap='viridis',
#                       linewidth=0)
    
    # Set the z axis limits
    #ax.set_zlim(node_v.min(),node_v.max())
    
    # Make the ticks looks pretty
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    
    # Include a colour bar
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    # Show the plot
    show()

plot_fem_solution(grid)
        
        
#timeit.timeit('plot_fem_solution(grid)')    
##    
##
#            

#
#
#
#












