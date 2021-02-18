#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:58:24 2019

@author: caglayantuna
"""


import siamxt
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
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew6.png')


#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)
imarray6=geoImToArray(Image6)

merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
tree = siamxt.MorphTreeAlpha(merged, Bc)

nodes=np.where(tree.node_array[1,:]==0)[0]

weights=parent_level_difference_nodes(tree,nodes)

lca=lowest_common_ancestor(tree,nodes)

cost=np.sum(tree.node_array[3,lca]/weights)