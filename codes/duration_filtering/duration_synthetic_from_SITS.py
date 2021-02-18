#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:35 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from imageio import imsave

from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
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
    res=np.abs(res)
    return res
def rmse_for_all(im,res,c):  
    gt=np.copy(im)
    r,col,b=gt.shape
    res[c==0]=0
    gt[c==0]=0

    result=np.zeros(b)
    for i in range(b):
        result[i]=np.sqrt(mean_squared_error(gt[:,:,i], res[:,:,i]))
    return result
def road_extraction(im):
    res=np.zeros(im.shape)
    res[im<105]=255
    res[im>160]=255
    return res
def accuracy(im,gt):
    gtnew=np.copy(gt)
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gtnew == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gtnew == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gtnew == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gtnew == 255)
    ppv=TP/(TP+FP)
    tpr=TP/(TP+FN)
    acc=(TP+FP)/(TP+TN+FN+FP)
    return ppv, tpr
def back_sub(gt):  
    r,c,b=gt.shape
    gt=np.array(gt,np.float)
    result=np.zeros([r,c],dtype=np.float)
    for i in range(b-1):
        result+=gt[:,:,i]-gt[:,:,i+1]   
    return np.abs(result)
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar1.png')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrcar6.png')
#im prepare
imarray1car=geoImToArray(Image1)[:,:,0]
imarray2car=geoImToArray(Image2)[:,:,0]
imarray3car=geoImToArray(Image3)[:,:,0]
imarray4car=geoImToArray(Image4)[:,:,0]
imarray5car=geoImToArray(Image5)[:,:,0]
imarray6car=geoImToArray(Image6)[:,:,0]

#data pleiades   
Imageref1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')
Imageref2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew2.png')
Imageref3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew3.png')
Imageref4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew4.png')
Imageref5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew5.png')
Imageref6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew6.png')
#im prepare
imarray1=geoImToArray(Imageref1)
imarray2=geoImToArray(Imageref2)
imarray3=geoImToArray(Imageref3)
imarray4=geoImToArray(Imageref4)
imarray5=geoImToArray(Imageref5)
imarray6=geoImToArray(Imageref6)

merged=np.dstack((imarray1car,imarray2car,imarray3car,imarray4car,imarray5car,imarray6car))
ref=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))


#max tree
Bc = np.ones((3,3,6), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)

#filtering
filtered=duration_filter(treemax,0)
identgt=merged-ref
identgt=np.sum(identgt,axis=2)
identgt[identgt>0]=255

rmse_tree=rmse_for_all(ref,filtered,identgt)

identified=merged-filtered
ident=np.sum(identified,axis=2)
ident[ident>0]=255
ppv1,tpr1=accuracy(ident,identgt)

res=back_sub(merged)
res[res<20]=0
res[res>=20]=255
ppv2,tpr2=accuracy(res,identgt)

reconslt=temp_interpolate(merged,identified)

rmse_recons=rmse_for_all(ref,reconslt,identgt)
#imsave('results/sythetic_from_SITS1.png',merged[:,:,0])
#imsave('results/sythetic_from_SITS2.png',merged[:,:,1])
#imsave('results/sythetic_from_SITS3.png',merged[:,:,2])
#imsave('results/sythetic_from_SITS4.png',merged[:,:,3])
#imsave('results/sythetic_from_SITS5.png',merged[:,:,4])
#imsave('results/sythetic_from_SITS6.png',merged[:,:,5])
identified[identified>0]=255
identified=255-identified

#imsave('results/sythetic_from_SITS_identified1.png',identified[:,:,0])
#imsave('results/sythetic_from_SITS_identified2.png',identified[:,:,1])
#imsave('results/sythetic_from_SITS_identified3.png',identified[:,:,2])
#imsave('results/sythetic_from_SITS_identified4.png',identified[:,:,3])
#imsave('results/sythetic_from_SITS_identified5.png',identified[:,:,4])
#imsave('results/sythetic_from_SITS_identified6.png',identified[:,:,5])

#imsave('results/sythetic_from_SITS_filtered1.png',filtered[:,:,0])
#imsave('results/sythetic_from_SITS_filtered2.png',filtered[:,:,1])
#imsave('results/sythetic_from_SITS_filtered3.png',filtered[:,:,2])
#imsave('results/sythetic_from_SITS_filtered4.png',filtered[:,:,3])
#imsave('results/sythetic_from_SITS_filtered5.png',filtered[:,:,4])
#imsave('results/sythetic_from_SITS_filtered6.png',filtered[:,:,5])