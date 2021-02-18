#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:23:38 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *


def parent_level_difference_nodes(tree,nodes):
    parent=tree.node_array[0,nodes]
    leveldiff=tree.node_array[2,nodes]-tree.node_array[2,parent]
    return leveldiff
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
tree = siamxt.MaxTreeAlpha(merged, Bc)

num_leaves=np.count_nonzero(tree.node_array[1,:]==0)

nodes=np.where(tree.node_array[1,:]==0)[0]

weights=parent_level_difference_nodes(tree,nodes)#weights for leaves 

weights=weights/np.sum(weights)#p

vertex_weights=parent_level_difference(tree)[1:] #weights for all
vertex_weights=weights/np.sum(vertex_weights)#q

index=np.where(weights)

tsd=np.sum(weights * np.log(weights / vertex_weights))