#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:35:07 2019

@author: caglayantuna
"""

import siamxt
import numpy as np
from functions.project_functions import *



def parent_attribute_difference(tree,attr):
    parent=tree.node_array[0,:]
    attrdiff=attr-attr[parent]
    attrdifference=np.zeros([255])
    for  i in range(255):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        attrdifference[i]=np.sum(attrdiff[nodes])
    return attrdifference
def plot_attr(attr):
    plt.figure()
    plt.ylabel('Difference of variances', fontsize=40)
    plt.xlabel('Level',fontsize=40)
    plt.plot(attr,'k')
    plt.show
def level_threshold(tree,threshold):
    nodes = np.where(threshold<=tree.node_array[2,:])[0]
    return nodes
def sum_of_nodes(mxt,node):
    a=np.zeros(mxt.shape)
    for i in range(node.size):
        a=a+mxt.recConnectedComponent(node[i],bbonly = False) 
    a[a>0]=255
    return a
def node_show(a):
    b=a.shape[2]
    for i in range(b):
       plt.figure()
       plt.imshow(a[:,:,i], cmap='gray')
       plt.show()
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#im prepare
im1=geoImToArray(Image1)
im2=geoImToArray(Image2)
im3=geoImToArray(Image3)

Bc = np.ones((3,3,3), dtype = bool)
merged=np.dstack([im1,im2,im3])

tree1 = siamxt.MaxTreeAlpha(merged, Bc)

attribute1=tree1.computeRR()
#attribute2=tree1.node_array[2,:] 

diff1=parent_attribute_difference(tree1,attribute1)
plot_attr(diff1)

gradient = diff1[0:-1] - diff1[1:]
level = np.argsort(gradient)[::-1][0]
node=level_threshold(tree1,level)

res=sum_of_nodes(tree1,node)
res=np.array(res,dtype=np.uint8)

node_show(res)