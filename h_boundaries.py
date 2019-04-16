#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:04:56 2018

@author: shilu
"""
#########################################################
# This module computes the values on boundaries, which
# will be later moved to right hand side to have 
# zero Dirichlet boundary condition
#########################################################

import numpy as np

from functions import Linear, Xlinear, Ylinear, Zero
from scipy import zeros, linalg




def hboundaries(alpha, h, n, nodes, intnodes, c, g1, g2, w):

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
            #print IDj, 'j'
            # For interior nodes
            if nodes[i][0] != 1 and nodes[i][0] != 0 and nodes[i][1] != 0 and nodes[i][1] != 1:
            
                IDi += 1
                #print IDi, 'i'
            
            # For boundary neighboured interior nodes
                
            
                # Left edge 
                if (intnodes[IDi][0] ==h) and (intnodes[IDi][1] != h) and (intnodes[IDi][1] != (n-2)*h) :
                    
                
                    h1[IDi] =  (c[IDj-1] +c[IDj-n-1])* h**2/float(12)+\
                               (- w[IDj-1])
                               
                    h2[IDi] = alpha* ( -g1[IDj-1])+\
                              (-2*w[IDj-1] - w[IDj-n-1])*h/float(6)
                              
                        
                    h3[IDi] = alpha* ( - g2[IDj-1])+\
                              (w[IDj-1] - w[IDj-n-1])*h/float(6)
                              
                    h4[IDi] = ( -c[IDj-1])-\
                              (-2*g1[IDj-1] - g1[IDj-n-1])*h/float(6)-\
                              (g2[IDj-1] - g2[IDj-n-1])*h/float(6)
                
                # Down edge               
                elif     (intnodes[IDi][1] == h)  and (intnodes[IDi][0] != h) and (intnodes[IDi][0] != (n-2)*h):  
                              
                        h1[IDi] =  (c[IDj-n]  + c[IDj-n-1])* h**2/float(12)+\
                                   (-w[IDj-n])
                                   
                        h2[IDi] = alpha* (-g1[IDj-n])+\
                                  (w[IDj-n] - w[IDj-n-1])*h/float(6)
                                  
                           
                        h3[IDi] = alpha*(- g2[IDj-n])+\
                                  (-2*w[IDj-n]- w[IDj-n-1])*h/float(6)
                                  
                        h4[IDi] = (- c[IDj-n])-\
                                  (g1[IDj-n] - g1[IDj-n-1])*h/float(6)-\
                                  (-2*g2[IDj-n]- g2[IDj-n-1])*h/float(6)
                                  
                                  
                 # Up edge                 
                elif   (intnodes[IDi][1] == (n-2)*h)  and (intnodes[IDi][0] != h) and (intnodes[IDi][0] != (n-2)*h):
                     
                     
                        h1[IDi] =  (c[IDj+n] + c[IDj+n+1])* h**2/float(12)+\
                                  (  -w[IDj+n])
                                   
                        h2[IDi] = alpha*(-g1[IDj+n])+\
                                  (-w[IDj+n] +w[IDj+n+1])*h/float(6)
                                  
                        #print IDi          
                        h3[IDi] = alpha*(-g2[IDj+n])+\
                                  (2*w[IDj+n]+ w[IDj+n+1])*h/float(6)
                                  
                        h4[IDi] = ( - c[IDj+n])-\
                                  ( - g1[IDj+n]+ g1[IDj+n+1])*h/float(6)-\
                                  (  2*g2[IDj+n]+ g2[IDj+n+1])*h/float(6)
                     
                     
                     
                # Right edge     
                elif (intnodes[IDi][0] == (n-2)*h ) and (intnodes[IDi][1] != h) and (intnodes[IDi][1] != (n-2)*h): 
                    
                    
                        h1[IDi] =  (c[IDj+1] + c[IDj+n+1])* h**2/float(12)+\
                                  (- w[IDj+1])
                                   
                        h2[IDi] = alpha* (- g1[IDj+1])+\
                                  (2* w[IDj+1] + w[IDj+n+1])*h/float(6)
                                  
                        #print IDi          
                        h3[IDi] = alpha* (- g2[IDj+1])+\
                                  (-  w[IDj+1]  + w[IDj+n+1])*h/float(6)
                                  
                        h4[IDi] = ( - c[IDj+1])-\
                                  (2* g1[IDj+1]+ g1[IDj+n+1])*h/float(6)-\
                                  ( - g2[IDj+1] + g2[IDj+n+1])*h/float(6)
                                  
                # Down left corner 
                elif (intnodes[IDi][0] ==h) and (intnodes[IDi][1] == h):
                    
                    h1[IDi] = (c[IDj-1] +c[IDj-n] +c[IDj-n-1])* h**2/float(12)+\
                              ( -w[IDj-1] -w[IDj-n])
                              
                    h2[IDi] = alpha* (-g1[IDj-1] - g1[IDj-n]) +\
                              (-2* w[IDj-1] - w[IDj-n-1] +w[IDj-n])*h/float(6)
                              
                    h3[IDi] = alpha* (-g2[IDj-1] - g2[IDj-n])+\
                              (w[IDj -1] - w[IDj -n-1] -2*w[IDj -n])*h/float(6)
                    
                    h4[IDi] = (-c[IDj-1] - c[IDj-n])-\
                              (-2* g1[IDj-1] - g1[IDj-n-1] + g1[IDj-n])*h/float(6)-\
                              (g2[IDj -1] -g2[IDj-n-1] -2* g2[IDj -n])*h/float(6)
            
               # Up left corner
                elif (intnodes[IDi][0] ==h) and (intnodes[IDi][1] == (n-2)*h):
                   
                    h1[IDi] = (c[IDj-1] +c[IDj+n] +c[IDj-n-1] +c[IDj+n+1])* h**2/float(12)+\
                              ( -w[IDj-1]  -w[IDj+n])
                              
                              
                    h2[IDi] = alpha* (-g1[IDj-1] - g1[IDj+n]) +\
                             (-2*w[IDj-1] -w[IDj+n] -w[IDj-n-1] + w[IDj+n+1])*h/float(6)
                             
                    h3[IDi] = alpha* (-g2[IDj-1] - g2[IDj+n])+\
                              (w[IDj-1] +2* w[IDj+n] - w[IDj-n-1]+ w[IDj+n+1])* h/float(6)
                              
                              
                    h4[IDi] = (-c[IDj-1] -c[IDj+n])-\
                              (-2*g1[IDj-1] -g1[IDj+n] +g1[IDj-n-1] + g1[IDj+n+1])*h/float(6)-\
                              (g2[IDj-1] + 2* g2[IDj+n] - g2[IDj-n-1]+g2[IDj+n+1])* h/float(6)
                    
                # Up right corner    
                elif ( intnodes[IDi][0] == (n-2)*h) and (intnodes[IDi][1] == (n-2)*h):
                    
                    h1[IDi] = (c[IDj+n] +c[IDj+n+1]+c[IDj+1])* h**2/float(12)+\
                              ( -w[IDj+1] -w[IDj+n])
                              
                              
                    h2[IDi] = alpha*(-g1[IDj+1] - g1[IDj+n])+\
                              ( -w[IDj+n] + 2* w[IDj+1] +w[IDj+n+1])*h/float(6)
                              
                    h3[IDi] = alpha*(-g2[IDj+1] - g2[IDj+n])+\
                              (2* w[IDj +n] - w[IDj+1] + w[IDj+n+1] )*h/float(6)
                              
                              
                    h4[IDi] = (-c[IDj+1] -c[IDj+n])-\
                              ( -g1[IDj+n] + 2* g1[IDj+1] +g1[IDj+n+1])*h/float(6)-\
                              (2* g2[IDj +n] - g2[IDj+1] + g2[IDj+n+1] )*h/float(6)
                              
                # Down right corner              
                elif ( intnodes[IDi][0] == (n-2)*h) and (intnodes[IDi][1] == h):
                    
                    h1[IDi] = (c[IDj-n] + c[IDj-n-1] +c[IDj+n+1]+c[IDj+1])* h**2/float(12)+\
                              (-w[IDj+1] -w[IDj-n])
                              
                    h2[IDi] = alpha* (-g1[IDj+1] -g1[IDj-n])+\
                              (2* w[IDj+1] + w[IDj-n] + w[IDj+n+1]  - w[IDj-n-1])*h/float(6)
                              
                    h3[IDi] = alpha* (-g2[IDj+1] -g2[IDj-n])+\
                              (-w[IDj+1] +w[IDj+n+1] -w[IDj-n-1] -2*w[IDj-n])*h/float(6)
                              
                    
                    h4[IDi] = (-c[IDj+1] -c[IDj-n])-\
                              (2*g1[IDj+1] +g1[IDj-n] -g1[IDj+n+1]  - g1[IDj-n-1])*h/float(6)-\
                              (-g2[IDj+1] +g2[IDj+n+1] -g2[IDj-n-1] -2*g2[IDj-n])*h/float(6)
                    
                    
        return hboundaries
    


            

                        
                        
                        
                        
              
    

        
    
        
        
    
     
