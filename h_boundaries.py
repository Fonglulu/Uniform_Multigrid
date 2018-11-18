#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:04:56 2018

@author: shilu
"""

import numpy as np

from functions import Linear, Xlinear, Ylinear, Zero
from scipy import zeros, linalg

i = 2

n= 2**i+1

# Find the spacing
h=1/float(n-1)

# Set the mesh grid, that is interior
#x1, y1 = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
x1, y1 = np.meshgrid(np.arange(0, 1+h, h), np.arange(0, 1+h, h))


intx, inty = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
nodes = np.vstack([x1.ravel(), y1.ravel()]).T
intnodes = np.vstack([intx.ravel(), inty.ravel()]).T



# Given the boundary information
c = Linear(x1, y1)
c = c.flatten()
g1 = Xlinear(x1, y1)
g1 = g1.flatten()
g2 = Ylinear(x1, y1)
g2 = g2.flatten()
w = Zero(x1,y1)
w = w.flatten()



def hboundaries(n, nodes, intnodes, c, g1, g2, w):

        hboundaries =  np.zeros((4, (n-2)**2, 1))
        
        hboundaries[0] = zeros([(n-2)**2,1])
        hboundaries[1] = zeros([(n-2)**2,1])
        hboundaries[2] = zeros([(n-2)**2,1])
        hboundaries[3] = zeros([(n-2)**2,1])
        
        
        h1 = hboundaries[0]
        h2 = hboundaries[1]
        h3 = hboundaries[2]
        h4 = hboundaries[3]
        
        
        
        
        IDi = -1
        IDj = -1
         #For all nodes
        for i in range(len(list(nodes))):
            
            IDj +=1
            print IDj, 'j'
            # For interior nodes
            if nodes[i][0] != 1 and nodes[i][0] != 0 and nodes[i][1] != 0 and nodes[i][1] != 1:
            
                IDi += 1
                print IDi, 'i'
            
            # For boundary neighboured interior nodes
                
            
                # Left edge
                if (intnodes[IDi][0] ==h) and (intnodes[IDi][1] != h) and (intnodes[IDi][1] != (n-2)*h) :
                    
                
                    h1[IDi] =  (c[IDj-1] +c[IDj-n-1])* h**2/float(12)+\
                               (- w[IDj-1])
                               
                    h2[IDi] = (-g1[IDj-1])+\
                              (2*w[IDj-1] + w[IDj-n-1])*h/float(6)
                              
                        
                    h3[IDi] = ( - g2[IDj-1])+\
                              (w[IDj-1] + w[IDj-n-1])*h/float(6)
                              
                    h4[IDi] = (- c[IDj-1])-\
                              (2*g1[IDj-1] + g1[IDj-n-1])*h/float(6)-\
                              (g2[IDj-1] + g2[IDj-n-1])*h/float(6)
                
                # Down edge               
                elif     (intnodes[IDi][1] == h)  and (intnodes[IDi][0] != h) and (intnodes[IDi][0] != (n-2)*h):  
                              
                        h1[IDi] =  (c[IDj-n]  + c[IDj-n-1])* h**2/float(12)+\
                                   (-w[IDj-n])
                                   
                        h2[IDi] = (g1[IDj-n])+\
                                  (w[IDj-n] + w[IDj-n-1])*h/float(6)
                                  
                           
                        h3[IDi] = (- g2[IDj-n])+\
                                  (2*w[IDj-n]+ w[IDj-n-1])*h/float(6)
                                  
                        h4[IDi] = (- c[IDj-n])-\
                                  (g1[IDj-n] + g1[IDj-n-1])*h/float(6)-\
                                  (2*g2[IDj-n]+ g2[IDj-n-1])*h/float(6)
                                  
                                  
                 # Up edge                 
                elif   (intnodes[IDi][1] == (n-2)*h)  and (intnodes[IDi][0] != h) and (intnodes[IDi][0] != (n-2)*h):
                     
                     
                        h1[IDi] =  (c[IDj+n] + c[IDj+n+1])* h**2/float(12)+\
                                  (4 * w[IDj] - w[IDj+1] - w[IDj-1] -w[IDj+n] -w[IDj-n])
                                   
                        h2[IDi] = (-g1[IDj+n])+\
                                  (-w[IDj+n] - w[IDj+n+1])*h/float(6)
                                  
                        print IDi          
                        h3[IDi] = (  -g2[IDj+n])+\
                                  (-2*w[IDj+n]- w[IDj+n+1])*h/float(6)
                                  
                        h4[IDi] = ( - c[IDj+n])-\
                                  ( - g1[IDj+n]- g1[IDj+n+1])*h/float(6)-\
                                  (  - 2*g2[IDj+n]- g2[IDj+n+1])*h/float(6)
                     
                     
                     
                # Right edge     
                elif (intnodes[IDi][0] == (n-2)*h ) and (intnodes[IDi][1] != h) and (intnodes[IDi][1] != (n-2)*h): 
                    
                    
                        h1[IDi] =  (c[IDj+1] + c[IDj+n+1])* h**2/float(12)+\
                                  (- w[IDj+1])
                                   
                        h2[IDi] = (- g1[IDj+1])+\
                                  (-2* w[IDj+1]- w[IDj+n+1])*h/float(6)
                                  
                        print IDi          
                        h3[IDi] = (- g2[IDj+1])+\
                                  (-  w[IDj+1]  - w[IDj+n+1])*h/float(6)
                                  
                        h4[IDi] = ( - c[IDj+1])-\
                                  (-2* g1[IDj+1]- g1[IDj+n+1])*h/float(6)-\
                                  ( - g2[IDj+1] - g2[IDj+n+1])*h/float(6)
                                  
                # Down left corner 
                elif (intnodes[IDi][0] ==h) and (intnodes[IDi][1] == h):
                    
                    h1[IDi] = (c[IDj-1] +c[IDj-n] +c[IDj-n-1])* h**2/float(12)+\
                              ( -w[IDj-1] -w[IDj-n])
                              
                    h2[IDi] = (-g1[IDj-1] - g1[IDj-n]) +\
                              (2* w[IDj-1] + w[IDj-n-1] +w[IDj-n])*h/float(6)
                              
                    h3[IDi] = (-g2[IDj-1] - g2[IDj-n])+\
                              (w[IDj -1] +w[IDj -n-1] +2*w[IDj -n])*h/float(6)
                    
                    h4[IDi] = (-c[IDj-1] - c[IDj-n])-\
                              (2* g1[IDj-1] + g1[IDj-n-1] +g1[IDj-n])*h/float(6)-\
                              (g2[IDj -1] +g2[IDj-n-1] +2* g2[IDj -n])*h/float(6)
            
               # Up left corner
                elif (intnodes[IDi][0] ==h) and (intnodes[IDi][1] == (n-2)*h):
                   
                    h1[IDi] = (c[IDj-1] +c[IDj+n] +c[IDj-n-1] +c[IDj+n+1])* h**2/float(12)+\
                              ( -w[IDj-1]  -w[IDj+n])
                              
                              
                    h2[IDi] = (-g1[IDj-1] -g1[IDj+n]) +\
                             (2*w[IDj-1] -w[IDj+n] +w[IDj-n-1] - w[IDj+1+1])*h/float(6)
                             
                    h3[IDi] = (-g2[IDj-1] - g2[IDj+n])+\
                              (w[IDj-1] -2* w[IDj+n] - w[IDj+n+1] +w[IDj-n-1])* h/float(6)
                              
                              
                    h4[IDi] = (-c[IDj-1] -c[IDj+n])-\
                              (2*g1[IDj-1] -g1[IDj+n] +g1[IDj-n-1] - g1[IDj+1+1])*h/float(6)-\
                              (g2[IDj-1] -2* g2[IDj+n] - g2[IDj+n+1] +g2[IDj-n-1])* h/float(6)
                    
                # Up right corner    
                elif ( intnodes[IDi][0] == (n-2)*h) and (intnodes[IDi][1] == (n-2)*h):
                    
                    h1[IDi] = (c[IDj+n] +c[IDj+n+1])* h**2/float(12)+\
                              ( -w[IDj+1] -w[IDj+n])
                              
                              
                    h2[IDi] = (-g1[IDj+1] - g1[IDj+n])+\
                              ( -w[IDj+n] - 2* w[IDj+1] -w[IDj+n+1])*h/float(6)
                              
                    h3[IDi] = (-g2[IDj+1] - g2[IDj+n])+\
                              (-2* w[IDj +n] - w[IDj+1] -w[IDj+n+1] )*h/float(6)
                              
                              
                    h4[IDi] = (-c[IDj+1] -c[IDj+n])-\
                              ( -g1[IDj+n] - 2* g1[IDj+1] -g1[IDj+n+1])*h/float(6)-\
                              (-2* g2[IDj +n] - g2[IDj+1] -g2[IDj+n+1] )*h/float(6)
                              
                # Down right corner              
                elif ( intnodes[IDi][0] == (n-2)*h) and (intnodes[IDi][1] == h):
                    
                    h1[IDi] = (c[IDj-n] + c[IDj-n-1] +c[IDj+n+1])* h**2/float(12)+\
                              (-w[IDj+1] -w[IDj-n])
                              
                    h2[IDi] = (-g1[IDj+1] -g1[IDj-n])+\
                              (-2*w[IDj+1] -w[IDj+n+1] +w[IDj-n] +w[IDj-n-1])*h/float(6)
                              
                    h3[IDi] = (-g2[IDj+1] -g2[IDj-n])+\
                              (-w[IDj+1] -w[IDj+n+1] -w[IDj-n-1] -2*w[IDj-n])*h/float(6)
                              
                    
                    h4[IDi] = (-c[IDj+1] -c[IDj-n])-\
                              (-2*g1[IDj+1] -g1[IDj+n+1] +g1[IDj-n] +g1[IDj-n-1])*h/float(6)-\
                              (-g2[IDj+1] -g2[IDj+n+1] -g2[IDj-n-1] -2*g2[IDj-n])*h/float(6)
                    
                    
        return hboundaries
    
    
hboundaries = hboundaries(n, nodes, intnodes, c, g1, g2, w)
        
                    
            

            

                        
                        
                        
                        
              
    

        
    
        
        
    
     