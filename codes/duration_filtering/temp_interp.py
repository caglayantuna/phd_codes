#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:24:51 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d


def rmse_for_all(im,res,c):  
    gt=np.copy(im)
    r,col,b=gt.shape
    res[c==0]=0
    gt[c==0]=0

    result=np.zeros(b)
    for i in range(b):
        result[i]=np.sqrt(mean_squared_error(gt[:,:,i], res[:,:,i]))
    return result

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
def create_gt():
   Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent1.png')
   Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent2.png')
   Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent3.png')
   Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent4.png')
   Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent5.png')
   Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent6.png')

   #im prepare
   imarray1=geoImToArray(Image1)[:,:,3]
   imarray2=geoImToArray(Image2)[:,:,3]
   imarray3=geoImToArray(Image3)[:,:,3]
   imarray4=geoImToArray(Image4)[:,:,3]
   imarray5=geoImToArray(Image5)[:,:,3]
   imarray6=geoImToArray(Image6)[:,:,3]
   merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
   return merged   

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


#min tree
Bc = np.ones((3,3,3), dtype=bool) 
tree = siamxt.MaxTreeAlpha(merged, Bc)

filteredmin=duration_filter(tree,0)

identified=merged-filteredmin


result=temp_interpolate(merged,identified)
gt=create_gt()


rmse_res=rmse_for_all(gt,result,identified)
