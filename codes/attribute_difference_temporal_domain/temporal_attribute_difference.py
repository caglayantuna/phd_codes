#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:51:36 2019

@author: caglayantuna
"""

    
import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

import functions.patternspectra




#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)

#imarray1=im_prepare(imarray1)
#imarray2=im_prepare(imarray2)
#imarray3=im_prepare(imarray3)


#max tree
Bc = np.ones((3,3), dtype=bool)
tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3, Bc)

#min tree
#tree1 = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)
#tree1.node_array[2,:]=imarray1.max()-tree1.node_array[2,:]
#tree2 = siamxt.MaxTreeAlpha(imarray2.max()-imarray2, Bc)
#tree2.node_array[2,:]=imarray2.max()-tree2.node_array[2,:]

a1=rect_image(tree1)
a2=rect_image(tree2)
#a3=rect_image(tree3)

c=a1-a2
#d=imarray3-imarray2
im_show(np.abs(c))
#im_show(d)

imsave("maxrectdiff.png",np.abs(c) )

