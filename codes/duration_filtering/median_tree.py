#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:44:27 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from imageio import imsave
def duration_image(tree):
    duration=time_duration(tree)
    dur_img=duration[tree.node_index]
    return dur_img
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
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)

#median tree
medim=np.abs(merged-np.median(merged))
medim=np.uint8(medim)
treemed=siamxt.MaxTreeAlpha(medim, Bc)


durmax=duration_image(treemax)
durmed=duration_image(treemed)
durmax[durmax==0]=255
durmed[durmed==0]=255
node_show(durmax)
node_show(durmed)
