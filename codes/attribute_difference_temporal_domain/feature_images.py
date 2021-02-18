#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:29:45 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

import functions.patternspectra




#data pleiades   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#data toulouse 2018  
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/toulouse/2018ndvimerged.tif')


#im prepare
imarray1=geoImToArray(Image1)
imarray1=np.array(imarray1,dtype=np.uint8)
#imarray2=geoImToArray(Image2)
#imarray3=geoImToArray(Image3)

#imarray1=im_prepare(imarray1)
#imarray2=im_prepare(imarray2)
#imarray3=im_prepare(imarray3)

#Bc = np.ones((3,3), dtype=bool)
#max tree
#tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)

#feature images
areaim=area_image_sits(imarray1)
heightim=height_image_sits(imarray1)

node_show(areaim)
node_show(heightim)

