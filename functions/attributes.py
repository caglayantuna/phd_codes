#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:54:58 2019

@author: caglayantuna
"""

def temp_stability(tree):
    nodesize=tree.node_array.shape[1]
    tempstability=np.zeros(nodesize)
    for i in range(nodesize):
        a=tree.recConnectedComponent(i,bbonly = False)
        area1=np.count_nonzero(a[:,:,0])
        area2=np.count_nonzero(a[:,:,1])
        area3=np.count_nonzero(a[:,:,2])
        tempstability[i]=np.std((area1,area2,area3))
    return tempstability
def area_image(tree):
      area_img = np.zeros(tree.node_index.shape, dtype=float)
      
      area=tree.node_array[3,:]
      area_img =area[tree.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
      return area_img
def mean_image(tree):
    mean=tree.computeNodeGrayAvg()
    mean_img =mean[tree.node_index]
    return mean_img
def parent_attribute_difference(tree,attr):
    parent=tree.node_array[0,:]
    attrdiff=attr-attr[parent]
    attrdifference=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        attrdifference[i]=np.sum(attrdiff[nodes])
    return attrdifference
def plot_attr(attr):
    plt.figure()
    plt.ylabel('Difference of variances', fontsize=40)
    plt.xlabel('Level',fontsize=40)
    plt.plot(attr,'k')
    plt.show
def stability_threshold(stability,threshold1,threshold2):
    node = np.where(np.logical_and(threshold1<=stability,stability <= threshold2))[0]    
    return node