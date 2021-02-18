#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:17:24 2020

@author: caglayantuna
"""

from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.insert(0,path)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions.project_functions import *
from sklearn.metrics import f1_score
from imageio import imwrite
def trainingsamples(im):
    classes=np.unique(gt)
    classes=classes[classes!=20]
    samples=np.zeros(classes.shape)
    for i in range(len(classes)):
        samples[i]=np.count_nonzero(im==i+1)
        
    return samples 


def classmeanstd(ref,im):
    r,c,b=im.shape
    classes=np.unique(ref)
    classes=classes[classes!=0]
    samples=np.zeros((classes.shape[0],b))
    for j in range(b):
        imcur=im[:,:,j]
        for i in range(len(classes)):
          samples[i,j]=np.mean(imcur[ref==classes[i]])
    np.savetxt('samplesanalysisdordogne.txt', (samples),fmt='%.2f',delimiter='&')     
    print(samples)      
    return samples
    
    
def visualize(ref,i):
    res=np.zeros((ref.shape[0],ref.shape[1],3))
    res[ref==1,:]=[255,255,0]
    res[ref==2,:]=[255,0,0]
    res[ref==3,:]=[0,255,255]
    res[ref==4,:]=[0,255,0]
    res[ref==5,:]=[255,0,255]
    res[ref==6,:]=[127,0,0]
    res[ref==7,:]=[0,127,0]
    res[ref==8,:]=[0,127,127]
    res= res.astype(np.uint8)
    imwrite('ref_dorgone'+str(i)+'.png',res)   
def visualize2(ref,i,Image):
    res=np.zeros((ref.shape[0],ref.shape[1],3))
    res[ref==1,:]=[255,255,0]
    res[ref==2,:]=[255,0,0]
    res[ref==3,:]=[0,255,255]
    res[ref==4,:]=[0,255,0]
    res[ref==5,:]=[255,0,255]
    res[ref==6,:]=[127,0,0]
    res[ref==7,:]=[0,127,0]
    res[ref==8,:]=[0,127,127]
    res[ref==9,:]=[0,0,127]
    res[ref==10,:]=[0,127,255]
    res[ref==11,:]=[210,105,0]
    res[ref==12,:]=[255,69,0]
    res= res.astype(np.uint8)
    #imwrite('ref_brittany'+str(i)+'.png',res)
    array_to_raster(res,Image,'ref_brittany'+str(i)+'.tif')
#Image = geoimread(path+'/dataset/land_cover_mapping/gt_raster_dordogne_cp.tif')
Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')


gt = geoImToArray(Image)
gt = gt.astype(np.uint8)


#Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_imposed.tif') 
Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif')      
imarray = geoImToArray(Image)


classanalysis=classmeanstd(gt,imarray)
#samples=trainingsamples(gt)


r, c = gt.shape
    
#vertical case
c_half=int(np.round(c/2))
gttrain=gt[:,0:c_half-100]
gttest=gt[:,c_half+100:-1]


samplesverttrain=trainingsamples(gttrain)
samplesverttest=trainingsamples(gttest)

visualize2(gttrain,1,Image)
visualize2(gttest,2,Image)

#horizontal calse case
r_half=int(np.round(r/2))
gttrain=gt[0:r_half-100,:]
gttest=gt[r_half+100:-1,:]


sampleshortrain=trainingsamples(gttrain)
sampleshortest=trainingsamples(gttest)

samples=np.dstack((samplesverttrain,samplesverttest,sampleshortrain,sampleshortest))


visualize2(gttrain,3,Image)
visualize2(gttest,4,Image)