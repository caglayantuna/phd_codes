#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:33:57 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
from skimage import exposure

def simplify_tree(tree):
    #simplify tree
    nodes=np.where(tree.node_array[1,:]>1)[0]
    nodeschildren=[]
    for i in range(nodes.shape[0]):
       children=tree.getChildren(nodes[i])
       nodeschildren.append(children)
    nodeschildren = np.array([ elem for singleList in nodeschildren for elem in singleList])
    nodespruned=np.ones(tree.node_array.shape[1])
    nodespruned[nodeschildren]=False
    nodespruned[0]=False #root
    nodespruned=np.array(nodespruned,dtype=int)
    #nodespruned=np.array(nodespruned, dtype=bool)
    newtree=tree.clone()
    a=newtree.prune(nodespruned)
    return a

def contrast_stretch(im):
   p2, p98 = np.percentile(im, (2, 98))
   img_rescale = exposure.rescale_intensity(res, in_range=(p2, p98))
   rescont=255*(img_rescale/img_rescale.max())
   return rescont

    
Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint8)


Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#max tree
tree= siamxt.MaxTreeAlpha(imarray, Bc)

#min tree
#tree = siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
#tree.node_array[2,:]=imarray.max()-tree.node_array[2,:]

#simplify tree
treenew=simplify_tree(tree)

res=treenew.getImage()
node_show(imarray)
node_show(res)

#area_image=tree.node_array[3,:][tree.node_index]

#area_imagesim=treenew.node_array[3,:][treenew.node_index]


# Contrast stretching after simplify
#rescont=contrast_stretch(res)
#imarraycont=contrast_stretch(imarray)
# Equalization after simplify
#res_eq = exposure.equalize_hist(res)
#node_show(255*res_eq)
#img_eq = exposure.equalize_hist(imarray)
#node_show(255*img_eq)

#node_show(rescont)
#node_show(imarraycont)


