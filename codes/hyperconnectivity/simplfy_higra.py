#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:18:48 2019

@author: caglayantuna
"""

from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
import higra as hg

    
Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint16)
imarray=imarray[:,:,0]
size=imarray.shape
graph = hg.get_4_adjacency_graph(size)
tree, altitudes = hg.component_tree_min_tree(graph, imarray)