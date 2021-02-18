#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:43:58 2019

@author: caglayantuna
"""


import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180601morbihanclip.tif')
imarray1=geoImToArray(Image1)

Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180621morbihanclip.tif')
imarray2=geoImToArray(Image2)

Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180701morbihanclip.tif')
imarray3=geoImToArray(Image3)

Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180711morbihanclip.tif')
imarray4=geoImToArray(Image4)

Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180731morbihanclip.tif')
imarray5=geoImToArray(Image5)

Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180805morbihanclip.tif')
imarray6=geoImToArray(Image6)

allfirstband=np.dstack((imarray1[:,:,0],imarray2[:,:,0],imarray3[:,:,0],imarray4[:,:,0],imarray5[:,:,0],imarray6[:,:,0]))

allsecondband=np.dstack((imarray1[:,:,1],imarray2[:,:,1],imarray3[:,:,1],imarray4[:,:,1],imarray5[:,:,1],imarray6[:,:,1]))

allthirdband=np.dstack((imarray1[:,:,2],imarray2[:,:,2],imarray3[:,:,2],imarray4[:,:,2],imarray5[:,:,2],imarray6[:,:,2]))

allforthband=np.dstack((imarray1[:,:,3],imarray2[:,:,3],imarray3[:,:,3],imarray4[:,:,3],imarray5[:,:,3],imarray6[:,:,3]))

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True


#min tree 1
tree1 = siamxt.MaxTreeAlpha(allfirstband.max()-allfirstband, Bc)
c=duration_filter(tree1,4)#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree1)
nodes=np.where(tempstabil>1000)[0]
result=sum_of_nodes(tree1,nodes)
node_show(result)

#min tree 2
tree2 = siamxt.MaxTreeAlpha(allsecondband.max()-allsecondband, Bc)
c=duration_filter(tree2,4)#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree2)
nodes=np.where(tempstabil>1000)[0]
result=sum_of_nodes(tree2,nodes)
node_show(result)

#min tree 3
tree3 = siamxt.MaxTreeAlpha(allthirdband.max()-allthirdband, Bc)
c=duration_filter(tree3,4)#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree3)
nodes=np.where(tempstabil>1000)[0]
result=sum_of_nodes(tree3,nodes)
node_show(result)

#min tree 4
tree4 = siamxt.MaxTreeAlpha(allforthband.max()-allforthband, Bc)
c=duration_filter(tree4,4)#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree4)
nodes=np.where(tempstabil>8000)[0]
result=sum_of_nodes(tree4,nodes)
node_show(result)