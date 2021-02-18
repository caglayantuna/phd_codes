#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:50:28 2019

@author: caglayantuna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:48:29 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

import functions.patternspectra


#synthetic 
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd1.png')
#imarray1=geoImToArray(Image1)[:,:,3]
#imarray1= imarray1.astype(np.uint8)


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')

#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
#imarray2=geoImToArray(Image2)
#imarray3=geoImToArray(Image3)




#tree building
Bc = np.ones((3,3), dtype=bool)
#min tree
#tree1 = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)
#tree1.node_array[3,:]=imarray1.max()-tree1.node_array[2,:]
#tree2 = siamxt.MaxTreeAlpha(imarray2.max()-imarray2, Bc)
#tree2.node_array[3,:]=imarray2.max()-tree2.node_array[2,:]

#tree3 = siamxt.MaxTreeAlpha(imarray3.max()-imarray3, Bc)
#max tree
tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)
#tree3 = siamxt.MaxTreeAlpha(imarray3, Bc)


ps1=area_weighted(tree1)
#ps2=area_weighted(tree2)
#ps3=area_weighted(tree3)


#area attribute
#area1=area_vector(tree1)
#area2=area_vector(tree2)
#area3=area_vector(tree3)
area1=mean_vector(tree1)
#area2=mean_vector(tree2)
#area3=mean_vector(tree3)

#area1=tree1.computeRR()
#area2=tree2.computeRR()
#area3=tree3.computeRR()

#shape attribute


shape1=tree1.computeRR()
#shape2=tree2.computeRR()
#shape3=tree3.computeRR()
#shape1=eccentricity_vector(tree1)
#shape2=eccentricity_vector(tree2)
#shape3=eccentricity_vector(tree3)
#shape1=mean_vector(tree1)
#shape2=mean_vector(tree2)
#shape3=mean_vector(tree3)

#2D pattern spectra
arealog_edges = np.logspace(np.log10(np.min(area1)), np.log10(np.max(area1) ), num=100)
shapelog_edges = np.logspace(np.log10(np.min(shape1)), np.log10(np.max(shape1) ), num=100)
res11,binedges11,binedges12,res=stats.binned_statistic_2d(area1,shape1,ps1,statistic='sum', bins=[100,100])
#res21,binedges21,binedges22,res=stats.binned_statistic_2d(area2,shape2,ps2,statistic='sum', bins=100)
#res31,binedges31,binedges32,res=stats.binned_statistic_2d(area3,shape3,ps3,statistic='sum', bins=100)

res11 = res11.astype(np.uint16)
#res21 = res21.astype(np.uint16)
#res31 = res31.astype(np.uint16)



#extract all bins first image ilki koordinata bakarak. olmuyor olmuyor araba cikmiyor. 
#yanklis cunku verdigi koordonatlar dogrud egil

#bins=np.nonzero(res11[90:,90:])
bins=np.where(res11>1000)
#bins=np.array(bins)+90
bins=np.array(bins)

nodes=get_nodes_from_bins(binedges11,binedges12,bins,area1, shape1)
result=reconstruct_nodes(tree1,nodes)


im_show(imarray1)
#im_show(imarray2)

#max
im_show(result)
#min
#im_show(imarray.max()-result)