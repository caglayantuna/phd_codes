#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:15:19 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
from skimage.util import random_noise
def lowest_common_ancestor(tree,nodes):
    ancestors=tree.getAncestors(np.int(nodes[0]))
    #ancestors=ancestors[ancestors!=nodes[0]]
    for i in range(nodes.shape[0]):
        anc=tree.getAncestors(np.int(nodes[i]))
        ancestors=np.intersect1d(ancestors,anc)
    return np.max(ancestors)
def dasgupta_cost_pairnew(tree):
    index=tree.node_index
    x,y=index.shape
    totalcost=0
    padded = np.pad(index, pad_width=1, mode='constant', constant_values=index.max()+1)
    for i in range(x):
        for j in range(y):
              node=padded[i+1,j+1]
              node1=padded[i,j+1]
              node2=padded[i,j]
              node3=padded[i+1,j]
              node4=padded[i+2,j]
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


Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint8)
imarray=imarray[200:500,100:400,3]

bc = np.ones((3,3), dtype=bool)

tree1= siamxt.MaxTreeAlpha(imarray, bc)

noisy = random_noise(imarray, mode='s&p',salt_vs_pepper=0.2)
noisy=np.uint8(255*noisy)

noisy = random_noise(imarray, mode='gaussian',var=0.001)
noisy=np.uint8(255*noisy)

noisy2 = random_noise(imarray, mode='gaussian',var=0.002)
noisy2=np.uint8(255*noisy2)

noisy3 = random_noise(imarray, mode='gaussian',var=0.003)
noisy3=np.uint8(255*noisy3)

tree2= siamxt.MaxTreeAlpha(noisy, bc)
tree3= siamxt.MaxTreeAlpha(noisy2, bc)
tree4= siamxt.MaxTreeAlpha(noisy3, bc)

cost1=dasgupta_cost_pairnew(tree1)
cost2=dasgupta_cost_pairnew(tree2)
cost3=dasgupta_cost_pairnew(tree3)
cost4=dasgupta_cost_pairnew(tree4)