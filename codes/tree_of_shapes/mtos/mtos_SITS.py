#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:15:53 2019

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
merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
        
        
        
tree=hg.component_tree_multivariate_tree_of_shapes_image2d(merged)

area=hg.attribute_area(tree)
filterednormal = hg.reconstruct_leaf_data(tree, area < 100)

im_show(filterednormal)
