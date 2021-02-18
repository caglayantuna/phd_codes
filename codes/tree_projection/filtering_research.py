#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:07:06 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy.misc import imsave

Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd1.png')
imarray1=geoImToArray(Image1)[:,:,3]


Bc = np.ones((3,3), dtype=bool)
tree= siamxt.MaxTreeAlpha(imarray1, Bc)
nodefirst=tree.node_array


nodesones=np.ones(tree.node_array.shape[1])
nodesones[5]=False
nodesones=np.array(nodesones,dtype=int)

nodefalse=np.zeros(tree.node_array.shape[1])
nodefalse[5]=True
nodefalse=np.array(nodefalse,dtype=int)

#contractDR
tree1=tree.clone()
tree1.contractDR(nodesones)
nodesdirectones=tree1.node_array

tree2=tree.clone()
tree2.contractDR(nodefalse)
nodesdirectfalse=tree2.node_array

#prune
tree3=tree.clone()
tree3.prune(nodesones)
nodespruneones=tree3.node_array

nodefalse[0]=0
nodefalse=np.array(nodefalse, dtype=bool)
tree4=tree.clone()
tree4.prune(nodefalse)
nodesprunefalse=tree4.node_array
