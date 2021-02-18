#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:25:01 2019

@author: caglayantuna
"""
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
import higra as hg

def parent_level_differencetos(tree,levels,area):
    parents=tree.parents()
    leveldiff=levels-levels[parents]
    multiply=np.multiply(leveldiff,area)
    return multiply
def pattern_spectra_nodestos(tree,binedges1,binedges2,bin1,bin2,attributevector1, attributevector2,altitudes):
    interval1=binedges1[bin1]
    interval2=binedges1[bin1+1]
    nodes1=find_node(interval1,interval2,attributevector1)
    
    interval1=binedges2[bin2]
    interval2=binedges2[bin2+1]
    nodes2=find_node(interval1,interval2,attributevector2)
    node=(np.intersect1d(nodes1, nodes2))

            
    deleted_nodes = np.ones(altitudes.shape, dtype=np.bool)
    deleted_nodes[node] = False
    a=hg.reconstruct_leaf_data(tree,altitudes,deleted_nodes)
    return a
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)


tree1, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray1)
tree2, altitudes2 = hg.component_tree_tree_of_shapes_image2d(imarray2)

area1 = hg.attribute_area(tree1)
area2=hg.attribute_area(tree2)
shape1=hg.attribute_compactness(tree1)
shape2=hg.attribute_compactness(tree2)

#pattern spectra
ps=parent_level_differencetos(tree1,altitudes,area1)
ps2=parent_level_differencetos(tree2,altitudes2,area2)
log_edges1 = np.logspace(np.log10(area1.min()), np.log10(area1.max()), num=100)
log_edges2 = np.logspace(np.log10(shape1.min()), np.log10(shape1.max()), num=100)
log_edges3 = np.logspace(np.log10(area2.min()), np.log10(area2.max()), num=100)
log_edges4 = np.logspace(np.log10(shape2.min()), np.log10(shape2.max()), num=100)
res11,binedges11,binedges12,res=stats.binned_statistic_2d(area1,shape1,ps,statistic='sum', bins=[log_edges1,log_edges2])
res21,binedges21,binedges22,res=stats.binned_statistic_2d(area2,shape2,ps2,statistic='sum', bins=[log_edges3,log_edges4])


#optimal transport
#bins1,bins2,otmat=ot_distance_all(res11,res21)
bins1,bins2,otmat=ot_distance_wozero(res11,res21)

#maximum distance
a=pattern_spectra_nodestos(tree1,binedges11,binedges12,bins1[0,0],bins1[0,1],area1, shape1,altitudes)
b=pattern_spectra_nodestos(tree2,binedges21,binedges22,bins2[0,0],bins2[0,1],area2, shape2,altitudes2)
