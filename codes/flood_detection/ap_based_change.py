#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:58:41 2019

@author: caglayantuna
"""

import siamxt
import numpy as np
import matplotlib.pyplot as plt
from project_functions import *

def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def ap_comparison(imarray1,imarray2,t1,t2,t3):
    #w, h, b = imarray1.shape
    #imarray1 = imarray1[:, :, 0].astype(np.uint8)
    #imarray2 = imarray2[:, :, 0].astype(np.uint8)
    Bc = np.zeros((3, 3), dtype=bool)
    Bc[1, :] = True
    Bc[:, 1] = True

    mxt1 = siamxt.MaxTreeAlpha(imarray1, Bc)
    result11 = attribute_area_filter(mxt1, t1)
    result12 = attribute_area_filter(mxt1, t2)
    result13 = attribute_area_filter(mxt1, t3)

    mxt1min = siamxt.MaxTreeAlpha(imarray1.max() - imarray1, Bc)
    result11min = attribute_area_filter(mxt1min, t1)
    result12min = attribute_area_filter(mxt1min, t2)
    result13min = attribute_area_filter(mxt1min, t3)

    mxt2 = siamxt.MaxTreeAlpha(imarray2, Bc)
    result21 = attribute_area_filter(mxt2, t1)
    result22 = attribute_area_filter(mxt2, t2)
    result23 = attribute_area_filter(mxt2, t3)

    mxt2min = siamxt.MaxTreeAlpha(imarray2, Bc)
    result21min = attribute_area_filter(mxt2min, t1)
    result22min = attribute_area_filter(mxt2min, t2)
    result23min = attribute_area_filter(mxt2min, t3)
    # B1=np.array(np.gradient(B1))

    AP1max = np.dstack((result11, result12, result13))
    AP1min = np.dstack((result11min, result12min, result13min))

    AP2max = np.dstack((result21, result22, result23))
    AP2min = np.dstack((result21min, result22min, result23min))

    # DAP1=np.dstack((result11-imarray1,result12-result11,result13-result12))
    # DAP2=np.dstack((result21-imarray2,result22-result21,result23-result22))
    maxdiff = np.absolute(AP1max - AP2max)
    mindiff = np.absolute(AP1min - AP2min)

    changemap= np.absolute(maxdiff + mindiff)
    changemapnew=changemap[:,:,0]
    #changemapnew = np.zeros([w, h])
    #changemapnew=changemapnew.astype(np.uint8)
    #a=57000 #first set 
    a=55000 #second set 
    changemapnew[changemapnew < a] = 0
    changemapnew[changemapnew > a] = 255
    changemapnew=changemapnew.astype(np.uint8)
    return changemapnew
def accuracy(im):
    #Image = geoimread('grountruhrasterized.png')
    Image = geoimread('images2/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
    
def imshow(im):
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()
#first dataset
#Image1 = geoimread('images/sar1DD4clipnew.tif')
#Image2 = geoimread('images/sarB757clipnew.tif')
#Image2= geoimread('images/sarC36Aclipnew.tif')
#Image2 = geoimread('images/sard22dclipnew.tif')

#second dataset
Image1 = geoimread('images2/sar1DD4clipsecond')
Image2 = geoimread('images2/sarB757clipsecond')
#Image2= geoimread('images2/sarC36Aclipsecond.tif')


imarray1=data_prepare(Image1)
#imarray2=data_prepare(Image2)
#imarray2=data_prepare(Image3)
imarray2=data_prepare(Image2)

Bc = np.zeros((3,3), dtype = bool)
Bc[1,:] = True
Bc[:,1] = True

change=ap_comparison(imarray1,imarray2,1000,2000,3000)
TP, TN, FP, FN=accuracy(change)

print((TP, TN, FP, FN )) 

Precision=(TP)/(TP+FP )
 
Recall=(TP)/(TP+FN )
 
F1=(2*Precision*Recall)/(Precision+Recall)


imshow(change)