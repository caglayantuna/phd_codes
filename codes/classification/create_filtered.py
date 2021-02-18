#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:09:27 2020

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
from skimage.morphology import erosion, dilation, opening, closing,reconstruction

from skimage.morphology import disk
def write_images(image,Image,filename):
    r,c,b=image.shape
    image=np.uint8(image)
     
    #array_to_raster(image[:,:,0],Image,'images/'+str(filename)+str(1)+'.tif')
    #array_to_raster(image[:,:,11],Image,'images/'+str(filename)+str(2)+'.tif')
    #array_to_raster(image[:,:,2*11],Image,'images/'+str(filename)+str(3)+'.tif')
    #array_to_raster(image[:,:,3*11],Image,'images/'+str(filename)+str(4)+'.tif')
    #array_to_raster(image[:,:,4*11],Image,'images/'+str(filename)+str(5)+'.tif')
    #array_to_raster(image[:,:,5*11],Image,'images/'+str(filename)+str(6)+'.tif')
    #imwrite('images/'+str(filename)+str(1)+'.png',image[2000:2500,2000:2500,0])
    #imwrite('images/'+str(filename)+str(2)+'.png',image[2000:2500,2000:2500,23])
    #imwrite('images/'+str(filename)+str(3)+'.png',image[2000:2500,2000:2500,2*23])
    #imwrite('images/'+str(filename)+str(4)+'.png',image[2000:2500,2000:2500,3*23])
    #imwrite('images/'+str(filename)+str(5)+'.png',image[2000:2500,2000:2500,4*23])
    #imwrite('images/'+str(filename)+str(6)+'.png',image[2000:2500,2000:2500,5*23])
    
    for i in range(b):
        imwrite(str(filename)+str(i+1)+'.png',image[2000:2500,2000:2500,i])
        
def mp_profile(im,Image):
    r,c,b=im.shape
    mp = np.zeros([r, c, 6 * b], dtype=float)
    thresholds=[3,5,7]
    for i in range(b):
        for j in thresholds: 
           eroded = erosion(im[:,:,i], disk(j))
           openingrec = np.uint8(reconstruction(seed=eroded, mask=im[:,:,i], method='dilation'))
           dilated = dilation(im[:,:,i], disk(j))
           closingrec = np.uint8(reconstruction(seed=dilated, mask=im[:,:,i], method='erosion'))      
           imwrite('mp_brittany_opening_date_'+str(i)+'_threshold_'+str(j)+'.png',openingrec[2000:2500,2000:2500])
           imwrite('mp_brittany_closing_date_'+str(i)+'_threshold_'+str(j)+'.png',closingrec[2000:2500,2000:2500])
def ap_cube_duration(im,Image):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [3,6,9]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)

    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        ap = duration_filter(mxt1, (t))
        mxt2 = mxtmin.clone()
        apmin= im.max()-duration_filter(mxt2, (t))
        for i in range(b):
              imwrite('sthdur_brittany_max_date_'+str(i+1)+'_threshold_'+str(t)+'.png',ap[2000:2500,2000:2500,i])
              imwrite('sthdur_brtittany_min_date_'+str(i+1)+'_threshold_'+str(t)+'.png',apmin[2000:2500,2000:2500,i])          
def ap_cube(im,Image):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [1000,10000,100000]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    #max tree profile

    for t in thresholds:
          mxt1 = mxt.clone()
          ap = attribute_area_filter(mxt1, (t))
          mxt2 = mxtmin.clone()
          apmin = im.max() -attribute_area_filter(mxt2, (t))
          for i in range(b):
              imwrite('sth_brittany_max_date_'+str(i+1)+'_threshold_'+str(t)+'.png',ap[2000:2500,2000:2500,i])
              imwrite('sth_brittany_min_date_'+str(i+1)+'_threshold_'+str(t)+'.png',apmin[2000:2500,2000:2500,i])          
     
def ap_marginal(imarray,Image):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.zeros((3, 3), dtype=bool)
    Bc[1,:]=1
    Bc[:,1]=1
    thresholds=[10,100,1000]
    attributes = np.zeros([r, c, 6 * b], dtype=float)

    for i in range(b):
        for j in thresholds:
           mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
           mxtmin=siamxt.MaxTreeAlpha(imarraymin[:, :, i], Bc)
           ap= attribute_area_filter(mxt, j)
           apmin = attribute_area_filter(mxt, j)
           imwrite('th_brittany_max_date_'+str(i+1)+'_threshold_'+str(j)+'.png',ap[2000:2500,2000:2500])
           imwrite('th_brittany_min_date_'+str(i+1)+'_threshold_'+str(j)+'.png',apmin[2000:2500,2000:2500])

def minmaxanalysis(im,b):
    for i in range(b):
        print(im[:,:,i].max())
        print(im[:,:,i].min())
        
def reference(ref):
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
    imwrite('images/ref_brittanyclipped.png',res)

#Image = geoimread(path+'/dataset/land_cover_mapping/gt_raster_dordogne_cp.tif')
Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')

gt = geoImToArray(Image)
gt = gt.astype(np.uint8)
#reference(gt[2000:2500,2000:2500])

#Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_int.tif') 
Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif')    

imarray = geoImToArray(Image)
#write_images(imarray,Image,'dordogne_date_')
#imwrite('images/origclipped.png',imarray[2000:2500,2000:2500,0])
r, c, b = imarray.shape
#vertical case
#c_half=int(np.round(c/2))
#imarray=imarray[:,0:c_half-100,:]    

#array_to_raster(imarray[:,:,0],Image,'orignew.tif')

areasth=ap_cube(imarray,Image)

#mp=mp_profile(imarray,Image)

#apdur=ap_cube_duration(imarray,Image)

#areath=ap_marginal(imarray,Image)


