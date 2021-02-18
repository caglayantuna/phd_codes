#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:19:08 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave


def border(im):
    im=255-im
    im[0,:,:]=0
    im[:,0,:]=0
    im[:,-1,:]=0
    im[-1,:,:]=0
    return im
    
#data   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/stability1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/stability2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/stability3.png')

#im prepare
imarray1=geoImToArray(Image1)[:,:,3]
imarray2=geoImToArray(Image2)[:,:,3]
imarray3=geoImToArray(Image3)[:,:,3]

imarray=np.dstack((imarray1,imarray2,imarray3))
#tree building
#Bc = np.ones((3,3,3), dtype=bool)
Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True


#min tree
tree1 = siamxt.MaxTreeAlpha(imarray, Bc)
duration_filter(tree1,1)
tempstabil=temp_stability_ratio(tree1)

#nodes=np.where(tempstabil<1.7)[0]
#result=sum_of_nodes(tree1,nodes)
result=stability_filter(tree1,0.1)
node_show(result)

node_show(border(imarray))