#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:57:53 2019

@author: caglayantuna
"""

import siamxt
import numpy as np
import matplotlib.pyplot as plt
from project import *

def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def find_leafnodes(mxt):
    leaf_nodes = np.where(mxt.node_array[1,:] == 0)[0]
    return leaf_nodes
def min_tree(im,bc):
    imarray=im.max()-im
    mxt = siamxt.MaxTreeAlpha(imarray,bc)
    return mxt

#firstdataset
#Image1 = geoimread('images/sar1DD4clipnew.tif')
#Image2 = geoimread('images/sarB757clipnew.tif')
#Image3= geoimread('images/sarC36Aclipnew.tif')
#Image4 = geoimread('images/sard22dclipnew.tif')
#seconddataset
Image1 = geoimread('images2/sar1DD4clipsecond')
Image2 = geoimread('images2/sarB757clipsecond')
Image3= geoimread('images2/sarC36Aclipsecond.tif')


imarray1=data_prepare(Image1)
imarray2=data_prepare(Image2)
imarray3=data_prepare(Image3)
#imarray4=data_prepare(Image4)


bc = np.zeros((3,3), dtype = bool)
bc[1,:] = True
bc[:,1] = True

#maxtree
tree1 = siamxt.MaxTreeAlpha(imarray1,bc)
tree2 = siamxt.MaxTreeAlpha(imarray2,bc)
tree3 = siamxt.MaxTreeAlpha(imarray3,bc)
#tree4 = siamxt.MaxTreeAlpha(imarray4,bc)

#mintree
treemin1 =min_tree(imarray1, bc)
treemin2 =min_tree(imarray2, bc)
treemin3 =min_tree(imarray3, bc)
#treemin4 =min_tree(imarray4, bc)


#amount of max nodes
shape1=tree1.node_array.shape[1]
shape2=tree2.node_array.shape[1]
shape3=tree3.node_array.shape[1]
#shape4=tree4.node_array.shape[1]

#amount of min nodes
shapemin1=treemin1.node_array.shape[1]
shapemin2=treemin2.node_array.shape[1]
shapemin3=treemin3.node_array.shape[1]
#shapemin4=treemin4.node_array.shape[1]

#sum of the area max
areasum1=tree1.node_array[3,:].sum()
areasum2=tree2.node_array[3,:].sum()
areasum3=tree3.node_array[3,:].sum()
#areasum4=tree4.node_array[3,:].sum()

#sum of the area min
areaminsum1=treemin1.node_array[3,:].sum()
areaminsum2=treemin2.node_array[3,:].sum()
areaminsum3=treemin3.node_array[3,:].sum()
#areaminsum4=treemin4.node_array[3,:].sum()

#total area
totalarea1=areasum1+areaminsum1
totalarea2=areasum2+areaminsum2
totalarea3=areasum3+areaminsum3
#totalarea4=areasum4+areaminsum4

#total nodes
totalnodes1=shape1+shapemin1
totalnodes2=shape2+shapemin2
totalnodes3=shape3+shapemin3
#totalnodes4=shape4+shapemin4

#mean gray value of the nodes
mean1=tree1.computeNodeGrayAvg().mean()
mean2=tree2.computeNodeGrayAvg().mean()
mean3=tree3.computeNodeGrayAvg().mean()
#mean4=tree4.computeNodeGrayAvg().mean()

#eccentricity of the nodes
eccentricity1=np.nansum(tree1.computeEccentricity()[2])
eccentricty2=np.nansum(tree2.computeEccentricity()[2])
eccentricty3=np.nansum(tree3.computeEccentricity()[2])
#eccentricty4=np.nansum(tree4.computeEccentricity()[2])

#leaf nodes max
leaf1=find_leafnodes(tree1)
leaf2=find_leafnodes(tree2)
leaf3=find_leafnodes(tree3)
#leaf4=find_leafnodes(tree4)

#leaf nodes min
leafmin1=find_leafnodes(treemin1)
leafmin2=find_leafnodes(treemin2)
leafmin3=find_leafnodes(treemin3)
#leafmin4=find_leafnodes(treemin4)

#stabil nodes max
stabil1=tree1.computeStabilityMeasure()
stabilnodes1= np.where(np.logical_and(stabil1<0.0001,0<stabil1))[0]
stabil2=tree2.computeStabilityMeasure()
stabilnodes2= np.where(np.logical_and(stabil2<0.0001,0<stabil2))[0]
stabil3=tree3.computeStabilityMeasure()
stabilnodes3= np.where(np.logical_and(stabil3<0.0001,0<stabil3))[0]
#stabil4=tree4.computeStabilityMeasure()
#stabilnodes4= np.where(np.logical_and(stabil4<0.0001,0<stabil4))[0]

#stabil nodes min
stabilmin1=treemin1.computeStabilityMeasure()
stabilminnodes1= np.where(np.logical_and(stabilmin1<0.0001,0<stabilmin1))[0]
stabilmin2=treemin2.computeStabilityMeasure()
stabilminnodes2= np.where(np.logical_and(stabilmin2<0.0001,0<stabilmin2))[0]
stabilmin3=treemin3.computeStabilityMeasure()
stabilminnodes3= np.where(np.logical_and(stabilmin3<0.0001,0<stabilmin3))[0]
#stabilmin4=treemin4.computeStabilityMeasure()
#stabilminnodes4= np.where(np.logical_and(stabilmin4<0.0001,0<stabilmin4))[0]

#total stabil nodes
totalstabil1=stabilnodes1.shape[0]+stabilminnodes1.shape[0]
totalstabil2=stabilnodes2.shape[0]+stabilminnodes2.shape[0]
totalstabil3=stabilnodes3.shape[0]+stabilminnodes3.shape[0]
#totalstabil4=stabilnodes4.shape[0]+stabilminnodes4.shape[0]

