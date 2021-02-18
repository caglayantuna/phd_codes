#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:00:45 2019

@author: caglayantuna
"""

from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
import higra as hg

#synthetic data 
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier1.png')
#imarray1=geoImToArray(Image1)[:,:,3]
#imarray1= imarray1.astype(np.uint8)

#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier2.png')
#imarray2=geoImToArray(Image2)[:,:,3]
#imarray2= imarray2.astype(np.uint8)

Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier3.png')
imarray3=geoImToArray(Image3)[:,:,3]
imarray3= imarray3.astype(np.uint8)


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

tree, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray1)
tree2, altitude2 = hg.component_tree_tree_of_shapes_image2d(imarray2)


#attribute difference
meanimage1= hg.reconstruct_leaf_data(tree, hg.attribute_mean_vertex_weights(tree,imarray1))
meanimage2= hg.reconstruct_leaf_data(tree2,hg.attribute_mean_vertex_weights(tree2,imarray2))
#im_show(meanimage1)
#im_show(meanimage2)
diff=np.abs(imarray1-imarray2)
im_show(diff)
im_show(np.abs(meanimage1-meanimage2))

imsave("tosmeandiff.png",np.abs(meanimage1-meanimage2))
