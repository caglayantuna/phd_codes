#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:40:19 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from sklearn.metrics import mean_squared_error

def rmse_for_all(gt,res):  
    r,c,b=gt.shape
    gt[gt>200]=0
    result=np.zeros(b)
    for i in range(b):
        result[i]=np.sqrt(mean_squared_error(gt[:,:,i], res[:,:,i]))
    return result

def back_sub(gt):  
    r,c,b=gt.shape
    gt=np.array(gt,np.float)
    result=np.zeros([r,c],dtype=np.float)
    for i in range(b-1):
        result+=gt[:,:,i]-gt[:,:,i+1]   
    return np.abs(result)
def create_gt(im):
    r,c,b=im.shape
    e=np.zeros([r,c])
    for i in range(b):
        coord=np.where(im[:,:,i]>199)
        coord=np.array(coord)
        e[coord[0],coord[1]]=255
    coord=np.where(im[:,:,5]==65)
    coord=np.array(coord)
    e[coord[0],coord[1]]=255
    return e
def accuracy(im,gt):
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    acc=(TP+FP)/(TP+TN+FN+FP)
    return acc
    
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier6.png')

#im prepare
imarray1=geoImToArray(Image1)[:,:,3]
imarray2=geoImToArray(Image2)[:,:,3]
imarray3=geoImToArray(Image3)[:,:,3]
imarray4=geoImToArray(Image4)[:,:,3]
imarray5=geoImToArray(Image5)[:,:,3]
imarray6=geoImToArray(Image6)[:,:,3]

merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
gt=create_gt(merged)
im_show(gt)

res=back_sub(merged)

res[res<40]=0
res[res>=40]=255
im_show(res)

acc=accuracy(res,gt)