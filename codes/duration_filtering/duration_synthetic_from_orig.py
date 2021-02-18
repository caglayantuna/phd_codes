#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:50:17 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
    
def rmse_for_all(im,res):
    gt=np.copy(im)
    r,c,b=gt.shape
    result=np.zeros(b)
    for i in range(b):
        result[i]=mean_squared_error(gt[:,:,i], res[:,:,i])
    return result
def back_sub(gt):  
    r,c,b=gt.shape
    gt=np.array(gt,np.float)
    result=np.zeros([r,c],dtype=np.float)
    for i in range(b-1):
        result+=gt[:,:,i]-gt[:,:,i+1]   
    return np.abs(result)
def road_extraction(im):
    res=np.zeros(im.shape)
    res[im<105]=255
    res[im>160]=255
    return res
def temp_interpolate(im,ident):
    r,c,b=im.shape
    res=np.copy(im)
    res=np.array(res,dtype=np.float)
    for i in range(b):
        coord=np.where(ident[:,:,i]!=0)
        coord=np.array(coord)
        for j in range(coord.shape[1]):
            coordx,coordy=coord[:,j]
            res[coordx,coordy,i]=np.nan
            idx_finite = np.isfinite(res[coordx,coordy,:])          
            x = np.arange(0, b)
            f_finite = interp1d(x[idx_finite], res[coordx,coordy,:][idx_finite],fill_value="extrapolate")
            res[coordx,coordy,:]= f_finite(x)

    return res
def accuracy(im):
    r,c=im.shape
    gt=np.zeros([r,c])
    gt[35:41,254:264]=255
    gt[130:140,62:68]=255
    gt[220:230,62:68]=255
    gt[173:179,183:193]=255
    gt[180:186,180:190]=255
    gt[180:186,13:23]=255
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    acc=(TP+TN)/(TP+TN+FP+FN)
    return acc


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')

#im prepare
imarray1=geoImToArray(Image1)

imarray2=np.copy(imarray1)

imarray3=np.copy(imarray1)

imarray4=np.copy(imarray1)

imarray5=np.copy(imarray1)

imarray6=np.copy(imarray1)

car=imarray1[13:19,256:266]
imarray1[35:41,254:264]=car
imarray2[130:140,62:68]=np.transpose(car)
imarray3[220:230,62:68]=np.transpose(car)
imarray4[173:179,183:193]=car
imarray5[180:186,180:190]=car
imarray6[180:186,13:23]=car

#imarray1 = ndimage.gaussian_filter(imarray1, sigma=1.4)
#imarray2 = ndimage.gaussian_filter(imarray2, sigma=1.2)
#imarray3 = ndimage.gaussian_filter(imarray3, sigma=1.7)
#imarray4 = ndimage.gaussian_filter(imarray4, sigma=1.4)
#imarray5 = ndimage.gaussian_filter(imarray5, sigma=1.8)
#imarray6 = ndimage.gaussian_filter(imarray6, sigma=1.9)
c=np.random.normal(0,1,[300,300])
c=np.array(c,dtype=np.uint8)
imarray1noise = np.uint8(imarray1+27*c)
imarray2noise = np.uint8(imarray2+10*c)
imarray3noise = np.uint8(imarray3+23*c)
imarray4noise = np.uint8(imarray4-12*c)
imarray5noise = np.uint8(imarray5+25*c)
imarray6noise = np.uint8(imarray6-15*c)



merged=np.dstack((imarray1noise,imarray2noise,imarray3noise,imarray4noise,imarray5noise,imarray6noise))
ref=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))


#max tree
Bc = np.ones((3,3,3), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)

#filtering
filtered=duration_filter(treemax,0)

#node_show(merged)
#node_show(filtered)
#node_show(merged-filtered)

res=rmse_for_all(ref,filtered)

identified=merged-filtered
ident=np.sum(identified,axis=2)
ident[ident>0]=255
acc1=accuracy(ident)

res=back_sub(merged)
res[res<20]=0
res[res>=20]=255
acc2=accuracy(res)


reconslt=temp_interpolate(merged,identified)

rmse_recons=rmse_for_all(ref,reconslt)
