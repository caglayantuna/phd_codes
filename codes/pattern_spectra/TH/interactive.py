#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:04:07 2019

@author: caglayantuna
"""
import siamxt
from functions.project_functions import *
from scipy import stats
from pylab import*

def find_bin_from_image(tree,im,attribute1,attribute2,res12,res13):
    im_show(im)
    y,x=ginput(1)[0]
    x=int(round(x))
    y=int(round(y))

    node = tree.node_index[x,y]
    att1=attribute1[node]
    att2=attribute2[node]
    binx= np.digitize(att1,res12)-1
    biny=np.digitize(att2,res13)-1
    return binx,biny
def find_bin_from_ps(ps):
    im_show(ps)
    y,x=ginput(1)[0]
    x=int(round(x))
    y=int(round(y))
    return x,y


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
#imarray2=geoImToArray(Image2)
#imarray3=geoImToArray(Image3)
#imarray1=im_prepare(imarray1)
#imarray2=im_prepare(imarray2)
#imarray3=im_prepare(imarray3)

#tree building
Bc = np.ones((3,3), dtype=bool)
#min tree
#tree1 = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2.max()-imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3.max()-imarray3, Bc)
#max tree
tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3, Bc)


ps1=area_weighted(tree1)
#ps2=area_weighted(tree2)
#ps3=area_weighted(tree3)


#area attribute
area1=area_vector(tree1)
#area2=area_vector(tree2)
#area3=area_vector(tree3)
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
#shape2=mean_vector(tree2)
#shape3=mean_vector(tree3)

#2D pattern spectra
res11,binedges11,binedges12,res=stats.binned_statistic_2d(area1,shape1,ps1,statistic='sum', bins=100)
#res21,binedges21,binedges22,res=stats.binned_statistic_2d(area2,shape2,ps2,statistic='sum', bins=100)
#res31,binedges31,binedges32,res=stats.binned_statistic_2d(area3,shape3,ps3,statistic='sum', bins=100)

#interactive ps
binx,biny=find_bin_from_ps(res11)
a=pattern_spectra_nodes(tree1,binedges11,binedges12,binx,biny,area1, shape1)
result=im_edit(a,tree1)

#interactive image to ps
binx,biny=find_bin_from_image(tree1,imarray1,area1,shape1,binedges11,binedges12)
a=pattern_spectra_nodes(tree1,binedges11,binedges12,binx,biny,area1, shape1)
result=im_edit(a,tree1)


im_show(result)