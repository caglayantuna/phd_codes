#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:13:26 2019

@author: caglayantuna
"""

import higra as hg
from functions.project_functions import *

#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew6.png')


#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)
imarray6=geoImToArray(Image6)


tree1, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray1)
tree2, altitudes2 = hg.component_tree_tree_of_shapes_image2d(imarray2)

tree_map=hg.tree_fusion_depth_map(tree1,tree2)