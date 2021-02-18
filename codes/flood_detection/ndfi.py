#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:28:36 2019

@author: caglayantuna
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from functions.project_functions import *


def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def ndfi(im1,im2):
    mean=meanSITS(im2)
    im1=np.array(im1,dtype=np.float)
    image=(mean-im1)/(mean+im1)
    image[image >= 0.58] = 255
    image[image < 0.58] = 0
    return image
def colored(im):
    #Image = geoimread('grountruhrasterized.png')
    
    Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    im=np.float64(im)
    res=im-gt
    r,c=im.shape
    c=np.zeros([r,c,3])
    c[res==-254]=[255,0,0]
    c[res==255]=[255,255,0]
    c[res==0]=[0,200,0]
    return c
def accuracy(im):
    #Image = geoimread('grountruhrasterized.png')
    
    Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    im=np.float64(im)
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
def imshow(result):
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()

#first dataset
#Image1 = geoimread('images/sar1DD4clipnew.tif')
#Image2 = geoimread('images/sarB757clipnew.tif')
#Image3= geoimread('images/sarC36Aclipnew.tif')
#Image4 = geoimread('images/sard22dclipnew.tif')

#second dataset   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata2.png')
Image3= geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata3.png')

imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)


#imarray4=data_prepare(Image2)

stacked=np.dstack((imarray2,imarray3))

result=ndfi(imarray1,stacked)


imshow(result)

bc = np.zeros((3,3), dtype = bool)
bc[1,:] = True
bc[:,1] = True
result=np.array(result,dtype=np.uint16)
mxt = siamxt.MaxTreeAlpha(result,bc)

areafiltered=attribute_area_filter(mxt,10)
TP, TN, FP, FN=accuracy(areafiltered)

print((TP, TN, FP, FN ))
imshow(areafiltered)

#imsave('ndfifirst.png',areafiltered)

Precision=(TP)/(TP+FP )
 
Recall=(TP)/(TP+FN )
 
F1=(2*Precision*Recall)/(Precision+Recall)