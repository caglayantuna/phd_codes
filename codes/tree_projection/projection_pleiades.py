#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:14:37 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave


def projected_tree(tree,t):
     node= np.unique(tree.node_index[:,:,t])
     nodesize=node.shape[0]
     tree_up=tree.clone()
     for i in range(nodesize):
        a=tree.recConnectedComponent(node[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree_up.node_array[3,node[i]]=area
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[node]=False
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)
     tree_up.node_index=tree_up.node_index[:,:,t]
     tree_up.node_array[0,0]=0
     return tree_up
 
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall6.png')

#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)
imarray5=geoImToArray(Image6)


merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
tree1 = siamxt.MaxTreeAlpha(merged, Bc)

bc = np.zeros((3,3), dtype=bool)
bc[1,:] = True
bc[:,1] = True

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#max tree
tree= siamxt.MaxTreeAlpha(merged, Bc)
#treenew=projected_tree(tree,2)
tree2d=siamxt.MaxTreeAlpha(merged[:,:,1], bc)

#min tree
#tree= siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
#tree1= siamxt.MaxTreeAlpha(imarray1.max()-imarray1, bc)

#projection
tree_pr=projected_tree(tree,1)

#filtering comparison
a=attribute_area_filter(tree_pr,50)
b=attribute_area_filter(tree2d,50)

#im_show(a)
#im_show(b)

#pattern spectra comparison

#area attribute
area1=area_vector(tree2d)
area2=area_vector(tree_pr)
#area1=mean_vector(tree1)
#area2=mean_vector(tree2)

#shape attribute
#shape1=eccentricity_vector(tree1)
#shape2=eccentricity_vector(tree2)

shape1=tree2d.computeRR()
shape2=tree_pr.computeRR()
#shape1=eccentricity_vector(tree2d)
#shape2=eccentricity_vector(tree_pr)
#shape1=mean_vector(tree2d)
#shape2=mean_vector(tree_pr)

ps1=area_weighted(tree2d)
ps2=area_weighted(tree_pr)
#2D pattern spectra
res11,res12,res13,res=stats.binned_statistic_2d(area1,shape1,ps1,statistic='sum', bins=100)
res21,res22,res23,res=stats.binned_statistic_2d(area2,shape2,ps2,statistic='sum', bins=100)

im_show(np.uint16(res11))
im_show(np.uint16(res21))