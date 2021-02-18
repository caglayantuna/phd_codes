#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:31:23 2020

@author: caglayantuna
"""

import numpy as np
import siamxt
from functions.project_functions import *

def parent_level_difference_nodes(tree,nodes):
    parent=tree.node_array[0,nodes]
    leveldiff=tree.node_array[2,nodes]-tree.node_array[2,parent]
    return leveldiff
def lowest_common_ancestor(tree,nodes):
    ancestors=tree.getAncestors(np.int(nodes[0]))
    #ancestors=ancestors[ancestors!=nodes[0]]
    for i in range(nodes.shape[0]):
        anc=tree.getAncestors(np.int(nodes[i]))
        ancestors=np.intersect1d(ancestors,anc)
    return np.max(ancestors)


    
def dasgupta_cost(tree):
    index=tree.node_index
    x,y=index.shape
    totalcost=0
    padded = np.pad(index, pad_width=1, mode='constant', constant_values=index.max()+1)
    for i in range(x):
        for j in range(y):
              node=padded[i+1,j+1]
              node1=padded[i,j+1]
              node2=padded[i+1,j+2]
              node3=padded[i+1,j]
              node4=padded[i+2,j+1]
              neighbors=np.stack((node1,node2,node3,node4))
              neighbors=neighbors[neighbors!=index.max()+1]
              cost=0
              for n in range(len(neighbors)):
                  pairs=np.array((node,neighbors[n]))
                  lca=lowest_common_ancestor(tree,pairs)
                  weights=np.abs(tree.node_array[2,pairs[0]]-tree.node_array[2,pairs[1]])
                  #costtemp=tree.node_array[3,lca]*weights
                  leaves=tree.getDescendants(int(lca)).size
                  costtemp=leaves*weights                 
                  cost+=costtemp
              totalcost+=cost
    return totalcost
def tsd(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    leaf_weights=parent_level_difference_nodes(tree,nodes)
    all_weights=np.sum(parent_level_difference(tree))
    p=leaf_weights/all_weights
    lca=lowest_common_ancestor(tree,nodes)
    node_weights=tree.node_array[2,nodes]-tree.node_array[2,lca]
    q=findSumofProduct(node_weights/all_weights)
    #q=tree.node_array[3,nodes]/tree.node_array[3,lca]
    index=np.where(p)
    res=np.sum(p[index] * np.log(p[index] / q[index]))
    return res
def euclidean_SITS(image):
    r, c, b = tuple(image.shape)
    ref=np.zeros(b)
    sh_image=np.zeros([r, c], dtype=float)
    for i in range(r):
        for j in range(c):
            sh_image[i, j] = np.sum(image[i,j,:]-ref)
    return sh_image
def projected_tree(tree,t):
     node= np.unique(tree.node_index[:,:,t])
     #node=np.array(np.where(np.logical_and(tree.node_array[13,:]>= t, tree.node_array[12,:]<= t)))[0]
     nodesize=node.shape[0]
     tree_up=tree.clone()
     for i in range(nodesize):
        a=tree.recConnectedComponent(node[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree_up.node_array[3,node[i]]=area
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[node]=False
     nodes[tree_up.node_array[3,:]==0]=True
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)
     tree_up.node_index=tree_up.node_index[:,:,t]
     tree_up.node_array[0,0]=0

     return tree_up
def projected_tree_sh(tree):
     r,c,b=tree.shape
     tree_up=tree.clone()

     duration_filter(tree_up,b-2)
     node= np.unique(tree_up.node_index)
     nodesize=node.shape[0]
     area=np.zeros(b)
     for i in range(nodesize):
        a=tree.recConnectedComponent(node[i],bbonly = False)
        c=np.sum(a,axis=2)
        c[c!=b]=0
        area=np.count_nonzero(c)
        tree_up.node_array[3,node[i]]=area
     nodespruned=np.where(tree_up.node_array[3,:]!=0)
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[nodespruned]=False
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)  
     tree_up.node_index=tree_up.node_index.min(axis=2)
     node= np.unique(tree_up.node_index)
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[node]=False
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)
     return tree_up
 
a=np.random.randint(4, size=(3, 3),dtype=np.uint8)

b=np.random.randint(4, size=(3, 3),dtype=np.uint8)

c=np.random.randint(4, size=(3, 3),dtype=np.uint8)
a=np.array([[1,	0,	1]
,[0,	0,	3]
,[3,	3,	3]],dtype=np.uint8)

b=np.array([[2,	2	,0],
[1,	0,	0],
[2,	2	,2]],dtype=np.uint8)

c=np.array([[1	,3,	2],
[2,	0,	0],
[1,	0	,1]],dtype=np.uint8)
    
#a=np.concatenate((a,a))
#b=np.concatenate((b,b))
#c=np.concatenate((c,c))
stacked=np.dstack((a,b,c))


bc = np.zeros((3,3), dtype=bool)
bc[1,:] = True
bc[:,1] = True

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#tree for each
tree_a=siamxt.MaxTreeAlpha(a, bc)
tree_b=siamxt.MaxTreeAlpha(b, bc)
tree_c=siamxt.MaxTreeAlpha(c, bc)

#space time tree
tree_st=siamxt.MaxTreeAlpha(stacked, Bc)

#th projection
a_pr=projected_tree(tree_st,0)
b_pr=projected_tree(tree_st,1)
c_pr=projected_tree(tree_st,2)

#sh projection
tree_pr_sh=projected_tree_sh(tree_st)
tree_sh=siamxt.MaxTreeAlpha(np.uint8(np.round(meanSITS(stacked))), bc)



#node arrays
node_array_st=tree_st.node_array
node_array_a=tree_a.node_array
node_array_b=tree_b.node_array
node_array_c=tree_c.node_array
node_array_a_pr=a_pr.node_array
node_array_b_pr=b_pr.node_array
node_array_c_pr=c_pr.node_array

node_array_pr_sh=tree_pr_sh.node_array
node_array_sh=tree_sh.node_array




#cost
cost_pr_c=dasgupta_cost(c_pr)
cost_c=dasgupta_cost(tree_c)
cost_pr_a=dasgupta_cost(a_pr)
cost_a=dasgupta_cost(tree_a)
cost_pr_b=dasgupta_cost(b_pr)
cost_b= dasgupta_cost(tree_b)