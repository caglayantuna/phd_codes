#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:45:35 2019

@author: caglayantuna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:26:53 2019

@author: caglayantuna
"""

#optimal transport

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
    node_show(a)
    return a
def reconstruct_nodes_tos(tree,altitudes,node):
        
    node=np.array(node,dtype=int)

    deleted_nodes = np.ones(altitudes.shape, dtype=np.bool)
    deleted_nodes[node] = False

    a, n_map =hg.simplify_tree(tree,deleted_nodes)
    newlev=altitudes[n_map]
    res=hg.reconstruct_leaf_data(a,newlev)
    return res
    
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
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





#extract all bins first image
bins=np.where(res11<20000)
bins=np.array(bins)

#extract all bins second image
bins2=np.where(res21>10000)
bins2=np.array(bins2)

#reconstruction
nodes=get_nodes_from_bins(binedges11,binedges12,bins2,area1, shape1)
result1=reconstruct_nodes_tos(tree1,altitudes,nodes)

#reconstruction
nodes2=get_nodes_from_bins(binedges21,binedges22,bins2,area2, shape2)
result2=reconstruct_nodes_tos(tree2,altitudes2,nodes2)

#normal filtering
area = hg.attribute_area(tree1)
filterednormal = hg.reconstruct_leaf_data(tree1, altitudes, area < 100)

im_show(imarray1)

#im_show(imarray2)


im_show(filterednormal)

im_show(result1)

#im_show(result2)
