#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:36:33 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from sklearn.metrics import mean_squared_error
from imageio import imsave

def rmse_for_all(im,res,c):  
    gtnew=np.copy(im)
    r,col,b=gt.shape
    res[c==0]=0
    gtnew[c==0]=0

    rmse=np.zeros(b)
    for i in range(b):
        rmse[i]=np.sqrt(mean_squared_error(gtnew[:,:,i], res[:,:,i]))
    return rmse

def back_sub(gt):  
    r,c,b=gt.shape
    gt=np.array(gt,np.float)
    result=np.zeros([r,c],dtype=np.float)
    for i in range(b-1):
        result+=gt[:,:,i]-gt[:,:,i+1]   
    return np.abs(result)
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
def accuracy(im,gt):
    
    gtnew=np.copy(gt)
    gtnew=np.sum(gtnew,axis=2)
    gtnew[gtnew>0]=255
    im[im==0]=1
    resul=im-gtnew
    TP=np.count_nonzero(resul == 0)/np.count_nonzero(gtnew == 255)
    TN=np.count_nonzero(resul == 1)/np.count_nonzero(gtnew == 0)
    FP=np.count_nonzero(resul == 255)/np.count_nonzero(gtnew == 0)
    FN=np.count_nonzero(resul == -254)/np.count_nonzero(gtnew == 255)
    ppv=TP/(TP+FP)
    tpr=TP/(TP+FN)
    acc=(TP+FP)/(TP+TN+FN+FP)
    return ppv, tpr

def colorized_detected(detected):
    im=np.copy(detected)
    im[im>0]=255
    im=255-im
    node_show(im)
    return im
def border(im):
    im[0,:]=0
    im[-1,:]=0
    im[:,-1]=0
    im[:,0]=0
    return im
    
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



gt=create_gt()

identified=merged-filteredmin
detected=colorized_detected(identified)
ident=np.sum(identified,axis=2)
ident[ident>0]=255
identgt=merged-gt
res=rmse_for_all(gt,filteredmin,identgt)



ppv,tpr=accuracy(ident,identgt)


#back suv
backres=back_sub(merged)

backres[backres>50]=255
ppvback,tprback=accuracy(backres,identgt)

imsave('detecetdsent1.png',border(detected[:,:,0]))
imsave('detecetdsent2.png',border(detected[:,:,1]))
imsave('detecetdsent3.png',border(detected[:,:,2]))
imsave('detecetdsent4.png',border(detected[:,:,3]))
imsave('detecetdsent5.png',border(detected[:,:,4]))
imsave('detecetdsent6.png',border(detected[:,:,5]))


imsave('outlier1.png',border(255-merged[:,:,0]))
imsave('outlier2.png',border(255-merged[:,:,1]))
imsave('outlier3.png',border(255-merged[:,:,2]))
imsave('outlier4.png',border(255-merged[:,:,3]))
imsave('outlier5.png',border(255-merged[:,:,4]))
imsave('outlier6.png',border(255-merged[:,:,5]))

imsave('durdiffsent1.png',border(255-filteredmin[:,:,0]))
imsave('durdiffsent2.png',border(255-filteredmin[:,:,1]))
imsave('durdiffsent3.png',border(255-filteredmin[:,:,2]))
imsave('durdiffsent4.png',border(255-filteredmin[:,:,3]))
imsave('durdiffsent5.png',border(255-filteredmin[:,:,4]))
imsave('durdiffsent6.png',border(255-filteredmin[:,:,5]))
