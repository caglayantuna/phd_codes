#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:42:25 2019

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
def project_tree_area(tree,t):    
     nodes= np.where(np.logical_and(tree.node_array[12,:]<=t,t<=tree.node_array[13,:]))[0]
     nodesize=nodes.shape[0]
     for i in range(nodesize):
        a=tree.recConnectedComponent(nodes[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree.node_array[3,nodes[i]]=area
     area_img=tree.node_array[3,:][tree.node_index][:,:,t]
     return area_img
def project_tree_mean_gray(tree,t):    
     nodes= np.where(np.logical_and(tree.node_array[12,:]<=t,t<=tree.node_array[13,:]))[0]
     nodesize=nodes.shape[0]
     for i in range(nodesize):
        a=tree.recConnectedComponent(nodes[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree.node_array[3,nodes[i]]=area
     area_img=tree.node_array[3,:][tree.node_index][:,:,t]
     return area_img
def area_image_th(imarray):
    Bc = np.zeros((3,3), dtype = bool)
    Bc[1,:] = True
    Bc[:,1] = True
    r, c, b = tuple(imarray.shape)
    area_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        area=mxt.node_array[3,:]
        area_img[:, :, i] =area[mxt.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
    return area_img

def projected_tree_th(tree,imarray):
    r, c, b = tuple(imarray.shape)
    area_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        area=project_tree_area(tree,i)
        area_img[:, :, i] =area
        #area_img=np.array(area_img,dtype=np.uint16)
    return area_img
#Image1 = geoimread('psdata/thirdps.tif')

Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint16)


    
bc = np.zeros((3,3), dtype=bool)
bc[1,:] = True
bc[:,1] = True

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#max tree
tree= siamxt.MaxTreeAlpha(imarray, Bc)
#treenew=projected_tree(tree,2)
tree2d=siamxt.MaxTreeAlpha(imarray[:,:,1], bc)

#min tree
#tree= siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
#tree1= siamxt.MaxTreeAlpha(imarray1.max()-imarray1, bc)



#area image sinflefrale
area2D=tree2d.node_array[3,:][tree2d.node_index]

#area images projection
tree_pr=projected_tree(tree,1)
area3Dpr=tree_pr.node_array[3,:][tree_pr.node_index]


meanimgpr=mean_image(tree_pr)
meanimg2d=mean_image(tree2d)
im_show(meanimgpr-meanimg2d)
#Area rate
ratio1=area3Dpr/area2D
ratio1[ratio1>100]=0
im_show(ratio1)

ratio1=area3Dpr/area2D
ratio1[ratio1<100]=0
im_show(ratio1)

ratio1=area3Dpr/area2D
ratio1[ratio1>2]=0
ratio1[ratio1<1.5]=0
im_show(ratio1)


#filtering comparison

a=attribute_area_filter(tree_pr,1000)
b=attribute_area_filter(tree2d,1000)

im_show(a-b)

