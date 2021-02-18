#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:00:08 2019

@author: caglayantuna
"""

import siamxt
import numpy as np
from project import *


def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def min_tree(im,bc):
    imarray=im.max()-im
    mxt = siamxt.MaxTreeAlpha(imarray,bc)
    return mxt
def bigareainleaf(mxt,leafnodes,area):
     node = np.where(mxt.node_array[3,leafnodes] >= area)[0]
     return node
def find_node_interval(mxt,interval,area):
    leaf_interval = np.where(np.logical_and(interval<=mxt.node_array[2,:],mxt.node_array[2,:] <= 65535))[0]
    nodeininterval=bigareainleaf(mxt,leaf_interval,area)
    node=leaf_interval[nodeininterval]
    return node
def parent_level_difference(tree):
    parent=tree.node_array[0,:]
    leveldiff=tree.node_array[2,:]-tree.node_array[2,parent]
    return leveldiff


#level difference  with parent and find the threshold
def level_difference_for_each_level(tree):
    leveldiff=parent_level_difference(tree)
    leveldifference=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        leveldifference[i]=np.sum(leveldiff[nodes])
    return leveldifference
def level_threshold(tree):
    levdiff=level_difference_for_each_level(tree)
    threshold=np.argmax(levdiff)
    return threshold

#variance different  with parent and find the threshold
def variance_for_each_level(tree):
    variance=tree.computeNodeGrayVar()
    varianceforlevel=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        varianceforlevel[i]=np.sum(variance[nodes])
    return varianceforlevel

#next three fucntions for parent variance difference for each level and find threshold
def parent_variance_difference(tree):
    variance=tree.computeNodeGrayVar()
    parent=tree.node_array[0,:]
    variancediff=variance[parent]-variance
    return variancediff
def variance_difference_for_each_level(tree):
    variancediff=parent_variance_difference(tree)
    variancedifference=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        variancedifference[i]=np.sum(variancediff[nodes])
    return variancedifference
def level_threshold_with_variance_differencre(tree):
    tree.node_array[2,:]=65535-tree.node_array[2,:]
    vardiff=variance_difference_for_each_level(tree)
    threshold=np.argmax(vardiff[0:5000])
    threshold=60000+threshold #yeni degisti
    return threshold

def figure_variance(tree):
    tree.node_array[2,:]=65535-tree.node_array[2,:]
    var=variance_difference_for_each_level(tree)
    var[0]=0
    plt.figure()
    plt.plot(var,'k')
    plt.ylabel('Variance')
    plt.xlabel('Level')
    plt.show()
def figure_level(tree):
    tree.node_array[2,:]=65535-tree.node_array[2,:]
    var=level_difference_for_each_level(tree)
    var[0]=0
    plt.figure()
    plt.plot(var,'k')
    plt.ylabel('Variance')
    plt.xlabel('Level')
    plt.show()
def imshow(result):
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()
def resultwithouttree(imarray1,imarray2,threshold):
    threshold=65535-threshold
    imarray1[imarray1<threshold]=255
    imarray1[imarray1>=threshold]=0
    imarray2[imarray2<threshold]=255
    imarray2[imarray2>=threshold]=0
    result=imarray1-imarray2
    result=result.astype(np.uint8)
    imshow(result)
   

#first dataset   
#Image1 = geoimread('images/sar1DD4clipnew.tif')
#Image2 = geoimread('images/sarB757clipnew.tif')
#Image2= geoimread('images/sarC36Aclipnew.tif')
#Image2 = geoimread('images/sard22dclipnew.tif')

#second dataset   
Image1 = geoimread('images2/sar1DD4clipsecond')
Image2 = geoimread('images2/sarB757clipsecond')
#Image2= geoimread('images2/sarC36Aclipsecond.tif')


imarray1=data_prepare(Image1)
imarray2=data_prepare(Image2)
#imarray3=data_prepare(Image3)
imarray=np.dstack([imarray1,imarray2])

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

treemin =min_tree(imarray, Bc)
#figure_variance(treemin)
figure_level(treemin)