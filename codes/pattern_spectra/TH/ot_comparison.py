#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:07:23 2019

@author: caglayantuna
"""

import siamxt
from scipy import stats
from scipy.misc import imsave
from pylab import*
import ot
from functions.project_functions import *

                   

#data pleiades   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')

#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')
#imarray1=geoImToArray(Image1)
#imarray2=geoImToArray(Image2)
#imarray1=np.array(imarray1/imarray1.max()*255,dtype=np.uint8)
#imarray2=np.array(imarray2/imarray2.max()*255,dtype=np.uint8)
#data morbihan
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image1)
imarray1=imarray[:,:,3]
imarray2=imarray[:,:,5]
imarray1=np.array(imarray1,dtype=np.uint8)
imarray2=np.array(imarray2,dtype=np.uint8)


#data rennes
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/change_detection/img1.png')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/change_detection/img2.png')


#im prepare
#imarray1=geoImToArray(Image1)[:,:,2]
#imarray2=geoImToArray(Image2)[:,:,2]
#imarray3=geoImToArray(Image3)
#imarray1=im_prepare(imarray1)
#imarray2=im_prepare(imarray2)
#imarray3=im_prepare(imarray3)


#tree building
Bc = np.ones((3,3), dtype=bool)
#min tree
tree1 = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)
tree2 = siamxt.MaxTreeAlpha(imarray2.max()-imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3.max()-imarray3, Bc)
#max tree
#tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3, Bc)


ps1=area_weighted(tree1)
ps2=area_weighted(tree2)
#ps3=area_weighted(tree3)


#area attribute
area1=area_vector(tree1)
area2=area_vector(tree2)
#rea3=area_vector(tree3)
#area1=mean_vector(tree1)
#area2=mean_vector(tree2)
#area3=mean_vector(tree3)

#area1=tree1.computeRR()
#area2=tree2.computeRR()
#area3=tree3.computeRR()

#shape attribute
#shape1=eccentricity_vector(tree1)
#shape2=eccentricity_vector(tree2)
#shape3=eccentricity_vector(tree3)

#shape1=tree1.computeRR()
#shape2=tree2.computeRR()
#shape3=tree3.computeRR()
#shape1=eccentricity_vector(tree1)
#shape2=eccentricity_vector(tree2)
#shape3=eccentricity_vector(tree3)
shape1=mean_vector(tree1)
shape2=mean_vector(tree2)
#shape3=mean_vector(tree3)

#2D pattern spectra
res11,res12,res13,res=stats.binned_statistic_2d(area1,shape1,ps1,statistic='sum', bins=100)
res21,res22,res23,res=stats.binned_statistic_2d(area2,shape2,ps2,statistic='sum', bins=100)
#res31,res32,res33,res=stats.binned_statistic_2d(area3,shape3,ps3,statistic='sum', bins=100)


#optimal transport
#bins1,bins2,otmat=ot_distance_all(res11,res21)
bins1,bins2,otmat=ot_distance_wozero(res11,res21)


p=1
a=pattern_spectra_nodes(tree1,res12,res13,bins1[p,0],bins1[p,1],area1, shape1)

b=pattern_spectra_nodes(tree2,res22,res23,bins2[p,0],bins2[p,1],area2, shape2)

im_show(a)

im_show(b)