#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:53:11 2020

@author: caglayantuna
"""

import siamxt
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from functions.project_functions import *


def parent_level_difference_nodes(tree,nodes):
    parent=tree.node_array[0,nodes]
    leveldiff=tree.node_array[2,nodes]-tree.node_array[2,parent]
    return leveldiff
def lowest_common_ancestor(tree,nodes):
    ancestors=tree.getAncestors(np.int(nodes[0]))
    for i in range(nodes.shape[0]):
        anc=tree.getAncestors(np.int(nodes[i]))
        ancestors=np.intersect1d(ancestors,anc)
    return np.max(ancestors)

def dasgupta_cost(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    weights=parent_level_difference_nodes(tree,nodes)
    lca=lowest_common_ancestor(tree,nodes)
    cost=np.sum(tree.node_array[3,lca]/weights)
    return cost
def tsd(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    leaf_weights=parent_level_difference_nodes(tree,nodes)
    all_weights=np.sum(parent_level_difference(tree))
    p=leaf_weights/all_weights
    lca=lowest_common_ancestor(tree,nodes)

    q=tree.node_array[3,nodes]/tree.node_array[3,lca]
    q[q==0]=1
    index=np.where(p)
    res=np.sum(p[index] * np.log(p[index] / q[index]))
    return res
def projected_tree_sh(tree):
     r,c,b=tree.shape
     duration_filter(tree,b-2)
     tree_up=tree.clone()
     node= np.unique(tree.node_index)
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
 
Image = geoimread(cwd+'/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint16)


bc = np.zeros((3,3), dtype=bool)
bc[1,:] = True
bc[:,1] = True

Bc = np.ones((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#max tree
tree= siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc) #space-time tree
treepr=projected_tree_sh(tree)
meanarray=np.uint16(dtw_image(imarray))
tree2d=siamxt.MaxTreeAlpha(meanarray.max()-meanarray, bc)

Bc = np.ones((3,3,6), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#new max tree
treenew= siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
treeprnew=projected_tree_sh(treenew)


#Tree quality cost
costpr=dasgupta_cost(treepr)
cost2d=dasgupta_cost(tree2d)
costprnew=dasgupta_cost(treeprnew)

#Tree quality tsd
tsdpr=tsd(treepr)
tsd2d=tsd(tree2d)
tsdprnew=tsd(treeprnew)