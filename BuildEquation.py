# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:19:15 2013

@author: stals

This class builds and stores the finite element matrix and load vector.
It also sets the Dirichlet boundary conditions.
"""

# Import appropriate information from other classes
from LinearPolynomial import LinearPolynomial
from Triangle import triangle_iterator, Interior_triangle_iterator
from grid.NodeTable import node_iterator, not_slave_node
from grid.ConnectTable import connect_iterator 
from grid.EdgeTable import endpt_iterator
from grid.NodeTable import node_iterator
from grid.Edge import DomainSet 
import numpy as np     
            
import math
from operator import itemgetter
    
    
#######################################################################
# Poisson_tri_integrate
#
# Evalute  a(u, v) where u and v are two polynomials,
# a(u, v) = int nabla u . nabla v dA and the integral is evaluated over
# the triangle (node1, node2, node3)
# In other words define the local stiffness matrix for Poisson's equation.
#
# It is assumed that all three nodes sit in a two dimension domain
#
# Input: LinearPolynomial poly1
#        LinearPolynomial poly2
#        Node node1
#        Node node2
#        Node node3
#
# Output: The edge between id1 and id2 as well as the edge 
# from id2 to id1 has been added to grid
#
#######################################################################
def Poisson_tri_integrate(poly1, poly2, node1, node2, node3):
    """Poisson model problem""" 
    
    # Import appropriate information from other classes
    from TriangleIntegrate import linear_integrate
    
    # Apply numerical quadrature routines to approximate the integrals
    local_stiffness_x = linear_integrate(poly1.dx(), poly2.dx(), 
                                      node1, node2, node3)
    local_stiffness_y = linear_integrate(poly1.dy(), poly2.dy(), 
                                        node1, node2, node3)
    
    return local_stiffness_x+local_stiffness_y


def TPS_tri_intergrateX(poly1, poly2, node1, node2, node3):
    
    from TriangleIntegrate import linear_integrate
    
    local_gx = linear_integrate(poly1, poly2.dx(), node1, node2, node3)
    
    return local_gx


def TPS_tri_intergrateY(poly1, poly2, node1, node2, node3):
    
    from TriangleIntegrate import linear_integrate
    
    local_gy = linear_integrate(poly1, poly2.dy(), node1, node2, node3)
    
    return local_gy
#######################################################################
# set_polynomial_linear_2D
#
# Find the linear polynomial whose value is 1 at node1 and 0 at
# nodes node2 and node3.
#
# It is assumed that all three nodes sit in a two dimension domain
#
#
# Input:
#        Node node1
#        Node node2
#        Node node3
#
# Output: LinearPolynomial poly
#
#######################################################################
def set_polynomial_linear_2D(node1, node2, node3):
    """Construct a linear polynomial"""
    
    # Import appropriate information from other classes
    from math import fabs
    
    # Check the coordinates are two dimensional
    assert node1.get_dim() == 2 and node2.get_dim() == 2 \
        and node3.get_dim() == 2, \
            "the triangle coordinates must be two dimensional"
            
    # Get the coordinates of the three vertices
    coord1 = node1.get_coord()
    coord2 = node2.get_coord()
    coord3 = node3.get_coord()
    
    # Break down the information to make the code easier to read
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    x3 = coord3[0]
    y3 = coord3[1]

    # Find h
    division = (y1-y2)*(x2-x3)-(y2-y3)*(x1-x2);
    
    # Avoid division by zero errors
    assert fabs(division) > 1.0E-12, "divide by zero in set_polynomial_linear_2D"

    # Find the polynomial coefficients
    poly = LinearPolynomial()
    poly.set_const((x3*y2 - y3*x2)/division)
    poly.set_x((y3-y2)/division)
    poly.set_y((x2-x3)/division)
    
    # Return the polynomial
    return poly

    
#######################################################################
# local_stiffness_linear_2D
#
# Find the local stiffness entries for the triangle (node1, node2, node3).
# The tri_integrate routine specifies the details of the current problem
# that is being solved. For example Poisson_tri_integrate defines the
# Poisson model problem
#
#
# Input:
#        function tri_integrate
#        Node node1
#        Node node2
#        Node node3
#
#
# Output: stiffness1, stiffness2 and stiffness3 which corresponds to the
# local stiffness matrix entries for node1-node1, node1-node2 and 
# node1-node3
#
#######################################################################
def local_stiffness_linear_2D(node1, node2, node3, tri_integrate):
    """Find the element stiffness matrix"""

    # Find the polynomials who's support is on the trinalge
    poly1 = set_polynomial_linear_2D(node1, node2, node3)
    poly2 = set_polynomial_linear_2D(node2, node3, node1)
    poly3 = set_polynomial_linear_2D(node3, node1, node2)

    # Evaluate the contribution to the local stiffness matrix
    local_stiffness1 = tri_integrate(poly1, poly1, node1, node2, node3)
    local_stiffness2 = tri_integrate(poly1, poly2, node1, node2, node3)
    local_stiffness3 = tri_integrate(poly1, poly3, node1, node2, node3)

    return local_stiffness1, local_stiffness2, local_stiffness3

#######################################################################
# local_load_linear_2D
#
# Find the local load entry for the triangle (node1, node2, node3).
# The right side function is given by rhs
#
#
# Input:
#        function rhs
#        Node node1
#        Node node2
#        Node node3
#
#
# Output: local load entry 
#
#######################################################################
def local_load_linear_2D(node1, node2, node3, rhs):
    """Find the element load vector"""
    
    # Import appropriate information from other classes
    from TriangleIntegrate import linear_func_integrate

    # Find a polynomial whose value is 1 at node1 and 0 at the other 
    # nodes
    poly1 = set_polynomial_linear_2D(node1, node2, node3)

    # Apply a numerical quadrature scheme to approximate the integral
    local_load = linear_func_integrate(rhs, poly1, node1, node2, node3)
                                      
    return local_load
    
#######################################################################
# set_slave_value
#
# If the node is joint to a boundary edge, then use the boundary
# function to assign a value to the node.
#
# Input: 
#        Grid grid
#        Node node
#
# Output: If a boundary function has been found, then the nodes value
# is equal to the boundary function evaluated at the node's coordinates.
# If no boundary function is found, then nothing happens.
#
#######################################################################
def set_slave_value(grid, node):
    """Assign the slave node a value"""
    
    # Loop through the edges joined to the node
    node_id = node.get_node_id()
    for endpt1 in endpt_iterator(grid, node_id): 
        
        # If the edge is a boundary edge
        if grid.get_location(node_id, endpt1) == DomainSet.boundary:
            
            # Evaluate the boundary function at the node coordinates
            bnd_func = grid.get_boundary_function(node_id, endpt1)
            node.set_value(bnd_func(node.get_coord()))
           

        
#######################################################################
# sum_load
#
# Add local_load to the current load value of node
#
#
# Input: Node node
#        float local_load
#
# Output: The load value of node has been increased by local_load
#
#######################################################################        
def sum_load(node, local_load):
    """Add local_load to the current load value"""
    
    # Find the current load value
    load = node.get_load()
    
    # Add local load
    node.set_load(load + local_load)

#######################################################################
# sum_stiffness
#
# Add local_stiff to the current matrix value corresponding to 
# node1, node2
#
#
# Input: Grid grid
#        Node node1
#        Node node2
#        float local_stiff
#
# Output: The matrix value has been increased by local_stiff
#
#######################################################################   
def sum_stiffness(grid, node1, node2, local_stiff):
    """Add local_stiff to the current matrix value"""
    
    # Get the node ids
    id1 = node1.get_node_id()
    id2 = node2.get_node_id()
    
    # Find the current stiffness value
    stiff = grid.get_matrix_value(id1, id2)[0]
    
    # Add local_stiff
    grid.set_matrix_value(id1, id2, np.array([stiff + local_stiff,\
                                              grid.get_matrix_value(id1,id2)[1], grid.get_matrix_value(id1,id2)[2],\
                                              grid.get_matrix_value(id1,id2)[3]]))
    
    
    
    

def sum_Aentry(grid, node1, node2, local_entry):
    """Add local_stiff to the current matrix value"""
    
    # Get the node ids
    id1 = node1.get_node_id()
    #id2 = node2.get_node_id()
    
    
    
    # Find the current stiffness value
    #Aentry = grid.get_matrix_value(id1, id2)[1]
    Aentry = grid.get_matrix_value(id1, node2)[1]
    
    # Add local_stiff
    grid.set_matrix_value(id1, node2, np.array([grid.get_matrix_value(id1,node2)[0],\
                                              Aentry+local_entry, grid.get_matrix_value(id1,node2)[2],\
                                              grid.get_matrix_value(id1,node2)[3]]))
    
    
    
def sum_G1(grid, node1, node2, local_g1):
    
    # Get the node ids
    id1 = node1.get_node_id()
    id2 = node2.get_node_id()
    
    # Find the current g1 value
    g1 = grid.get_matrix_value(id1, id2)[2]
    
    # Add local_g1
    grid.set_matrix_value(id1, id2, np.array([grid.get_matrix_value(id1,id2)[0],\
                                              grid.get_matrix_value(id1,id2)[1], \
                                              g1 +local_g1, grid.get_matrix_value(id1,id2)[3]]))
    
    
def sum_G2(grid, node1, node2, local_g2):

    # Get the node ids
    id1 = node1.get_node_id()
    id2 = node2.get_node_id()
    
    # Find the current g2 value
    g2 = grid.get_matrix_value(id1, id2)[3]
    
    # Add local_g2
    grid.set_matrix_value(id1, id2, np.array([grid.get_matrix_value(id1,id2)[0],\
                                              grid.get_matrix_value(id1,id2)[1], grid.get_matrix_value(id1,id2)[2], \
                                              g2 +local_g2]))
    
    
    

def NotIn_triangle(node1, node2, node3, data):
    """ This routine check if a data point sits inside(inclduing boundary)
    of the triangle identified by the given three nodes"""
    
    N1 = node1.get_coord()
    #print N1
    N2 = node2.get_coord()
    #print N2
    N3 = node3.get_coord()
    #print N3
    
    # Caluculate Barycentric coordinates
    Dominator = ((N2[1] - N3[1])*(N1[0] - N3[0]) + (N3[0] - N2[0])*(N1[1] - N3[1]))
    #print Dominator
    

    s = ((N2[1] - N3[1])*(data[0] - N3[0]) + (N3[0] - N2[0])*(data[1] - N3[1]))/float(Dominator)
    
    
    #print s
    
    t = ((N3[1] - N1[1])*(data[0] - N3[0]) + (N1[0] - N3[0])*(data[1] - N3[1]))/float(Dominator) 
    
    #print s, t, (1-s-t)
    
    return (s< 0.0 or  t< 0.0 or ((1.-s-t)< 0.0))


def NIn_triangle(node1, node2, node3, data):
    
    basis1 = set_polynomial_linear_2D(node1, node2, node3)
                    
    basis2 = set_polynomial_linear_2D(node2, node1, node3)
                    
    basis3 = set_polynomial_linear_2D(node3, node2, node1)
    
    value1 = basis1.eval(data[0], data[1])
    
    value2 = basis2.eval(data[0], data[1])
    
    value3 = basis3.eval(data[0], data[1])
    
    return (value1 >=0.0 and value2>=0.0 and value3 >=0.0)
    
    
    


#######################################################################
# build_equation_linear_2D
#
# Build the system of equations to solve the current model problem
#
# tri_integrate is a function that returns the local stiffness matrix
# values on the current model problem. See Poisson_tri_integrate for an
# example. rhs_function defines the right hand side function. It is 
# assumed that grid contains the nodes and the edges, but not necessarily
# the connections. If the connections are not in the grid, they will be
# added so that the matrix corresponding to linear basis functions can
# be stored. It is also assumed that the boundary functions are 
# stored with the edges, these boundary functions are used to set the
# slave nodes. 
#
# Input: Grid grid
#        function tri_integrate
#        function rhs_function
#
# Output: The matrix values are stored in the connection table
#         The load values (dependent on rhs_function) are stored in the
# load fields of the nodes.
#         Any slave nodes are assigned a value given by the boundary
# functions stored with the edges
#
#######################################################################   
def build_equation_linear_2D(grid, tri_integrate, rhs_function):
    """Define the stiffness matrix and load vector"""
    
    # Set the values for the slave nodes and initialise the load value 
    # to 0
    for node in node_iterator(grid):
        node.set_load(0.0)
        if (node.get_slave()):
            set_slave_value(grid, node)
        
    # Add the matrix connections for linear basis functions and 
    # initialise to zero
    for node in node_iterator(grid):
        node_id = node.get_node_id()
        if not grid.is_connection(node_id, node_id):
            grid.add_connection(node_id, node_id)
        grid.set_matrix_value(node_id, node_id, np.array([0.0,0.0, 0.0, 0.0]))
        for endpt1 in endpt_iterator(grid, node.get_node_id()):
            if not grid.is_connection(node_id, endpt1):
                grid.add_connection(node_id, endpt1)
            grid.set_matrix_value(node_id, endpt1, np.array([0.0, 0.0, 0.0, 0.0]))

                    
    # Evalue that A matrix and rhs vector
    
    # Loop over the triangles
    
    for tri in triangle_iterator(grid):
        #print(tri[0].get_node_id()._id_no)
        if tri[1].get_node_id() < tri[2].get_node_id():
            
            # Find the local stiffness entries
            stiff1, stiff2, stiff3 \
                = local_stiffness_linear_2D(tri[0], tri[1], tri[2],
                                            tri_integrate)
                
            

            # Find the local load entries
            local_load = local_load_linear_2D(tri[0], tri[1], tri[2], 
                                            rhs_function)
            
            # Add in the contributions from the current triangle
            sum_load(tri[0], local_load)
            sum_stiffness(grid, tri[0], tri[0], stiff1)
            sum_stiffness(grid, tri[0], tri[1], stiff2)
            sum_stiffness(grid, tri[0], tri[2], stiff3)
                      
            
            
    
    

def build_matrix_fem_2D(grid, tri_integrate1, tri_integrate2, tri_integrate3, X, Y):
    
#    coordx = X.flatten()
#    coordy = Y.flatten()
#    Coord = zip(coordx, coordy)

    
    """ This routine builds the A matrix G1 and G2 matrix and stiffness, and 
    store it on grid.
    """
    
    

        


        
    # Add the matrix connections for linear basis functions and 
    # initialise to zero
    for node in node_iterator(grid):
        node_id = node.get_node_id()
        if not grid.is_connection(node_id, node_id):
            grid.add_connection(node_id, node_id)
        grid.set_matrix_value(node_id, node_id, np.array([0.0, 0.0, 0.0, 0.0]))
        for endpt1 in endpt_iterator(grid, node.get_node_id()):
            if not grid.is_connection(node_id, endpt1):
                grid.add_connection(node_id, endpt1)
            grid.set_matrix_value(node_id, endpt1, np.array([0.0, 0.0, 0.0, 0.0]))

                    
    # Evalue that A matrix and rhs vector
    
    # Loop over the triangles
    
    for tri in triangle_iterator(grid):
        #print(tri[0].get_node_id()._id_no)
        if tri[1].get_node_id() < tri[2].get_node_id():
            
            # Find the local stiffness entries
            stiff1, stiff2, stiff3 \
                = local_stiffness_linear_2D(tri[0], tri[1], tri[2],
                                            tri_integrate1)
                
                
            # Add in the contributions from the current triangle
            #sum_load(tri[0], local_load)
            sum_stiffness(grid, tri[0], tri[0], stiff1)
            sum_stiffness(grid, tri[0], tri[1], stiff2)
            sum_stiffness(grid, tri[0], tri[2], stiff3)
            
            
            XG1, XG2, XG3 \
                = local_stiffness_linear_2D(tri[0], tri[1], tri[2],
                                            tri_integrate2)
                
            sum_G1(grid, tri[0], tri[0], XG1)
            sum_G1(grid, tri[0], tri[1], XG2)
            sum_G1(grid, tri[0], tri[2], XG3)
            
            
            YG1, YG2, YG3 \
                = local_stiffness_linear_2D(tri[0], tri[1], tri[2],
                                            tri_integrate3)
                
            sum_G2(grid, tri[0], tri[0], YG1)
            sum_G2(grid, tri[0], tri[1], YG2)
            sum_G2(grid, tri[0], tri[2], YG3)
            
                

               
                                            
            # Find the local load entries
            #local_load = local_load_linear_2D(tri[0], tri[1], tri[2], 
                                            #rhs_function)
    print 'setup'        

            
     #Store A matrix for whole grid nodes        
#    for tri in triangle_iterator(grid):
#    
#    
#        if (tri[1].get_node_id() < tri[2].get_node_id()) :
#            
#            basis1 = set_polynomial_linear_2D(tri[0], tri[1], tri[2])
#            
#            basis2 = set_polynomial_linear_2D(tri[1], tri[0], tri[2])
#            
#            basis3 = set_polynomial_linear_2D(tri[2], tri[1], tri[0])
#            
#    
#            
#            for i in Coord:
#            
#                if NIn_triangle(tri[0] , tri[1], tri[2], i):
#                
#                
#                        ii = basis1.eval(i[0], i[1]) * basis1.eval(i[0], i[1])
#                        
#                        sum_Aentry(grid, tri[0], tri[0], ii)
#            
#                        ij = basis1.eval(i[0], i[1]) * basis2.eval(i[0], i[1])
#                        
#                        sum_Aentry(grid, tri[0], tri[1], ij)
#                        
#                        ik = basis1.eval(i[0], i[1]) * basis3.eval(i[0], i[1])
#                        
#                        sum_Aentry(grid, tri[0], tri[2], ik)
#                        
    
    Nl=[]
    for node in (not_slave_node(grid)):
        
        Nl.append([node,node.get_node_id()])
        
    Nl=sorted(Nl, key = itemgetter(1))
    
    Nl=[node[0] for node in Nl]
            
    for node in Nl:
        node.set_value(Nl.index(node))
        
    h = 1 /float(int(math.sqrt(len(Nl)))+1)
    print h, 'h'
        
    for node in node_iterator(grid):
    
    
         for endpt in connect_iterator(grid, node.get_node_id()):
             
             #if not grid.get_slave(endpt): 
             
             
                     if (node.get_value() - grid.get_value(endpt)) == 0:
                         
                          sum_Aentry(grid, node, endpt, 6* h**2/float(12))
                        
                     elif abs(node.get_value() - grid.get_value(endpt)) ==1:
                        
                          sum_Aentry(grid, node, endpt, h**2 / float(12))
                        
                     elif abs(node.get_value() - grid.get_value(endpt)) == int(math.sqrt(len(Nl))):
                        
                          sum_Aentry(grid, node, endpt, h**2/float(12))
                        
                     elif abs(node.get_value() - grid.get_value(endpt)) == (int(math.sqrt(len(Nl))) +1):
                        
                          sum_Aentry(grid, node, endpt , h**2/float(12))
#                          
    print 'setAup'
             
             
            
            
                        
                
                        
                        
    
                        





    










