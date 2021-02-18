#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:43:32 2019

@author: caglayantuna
"""
from functions.project_functions import *
import siamxt

def ap_single(im,r,c):
    immin=im.max()-im
    Bc = np.ones((3,3),dtype = bool)
    thresholds =  [100,200,300]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,6))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt2 = mxt.clone()
        a= attribute_area_filter(mxt2, (t))
        a=np.reshape(a,[a.shape[0],a.shape[1],1])
        ap[:,:,0+i:1+i]=a
        i+=1
    #min tree profile
    i=0    
    for t in thresholds:
        mxt2 = mxtmin.clone()
        a=  im.max() -attribute_area_filter(mxt2, (t))
        a=np.reshape(a,[a.shape[0],a.shape[1],1])
        ap[:,:,3+i:4+i] = a
        i+=1 
    im=np.reshape(im,[im.shape[0],im.shape[1],1])
    approfile= np.concatenate((ap,im),axis=2)
    return approfile
def ap_sth(im,r,c,b):
    immin=im.max()-im
    Bc = np.ones((3,3,3),dtype = bool)
    thresholds =  [300,600,900]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt2 = mxt.clone()
        ap[:,:,0+i:b+i] = attribute_area_filter(mxt2, (t))
        i+=b
    #min tree profile
    i=0    
    for t in thresholds:
        mxt2 = mxtmin.clone()
        ap[:,:,3*b+i:4*b+i] = im.max() -attribute_area_filter(mxt2, (t))
        i+=b    
    approfile= np.concatenate((ap,im),axis=2)
    return approfile
def ap_th(imarray,r,c,b):
    imarraymin=imarray.max()-imarray
    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 7 * b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mxtmin=siamxt.MaxTreeAlpha(imarraymin[:, :, i], Bc)

        attributes[:, :, i] = attribute_area_filter(mxt, (100))
        attributes[:, :, b + i] = attribute_area_filter(mxt, (200))
        attributes[:, :, 2 * b + i] = attribute_area_filter(mxt, (300))
        attributes[:, :, 3 * b+i] = attribute_area_filter(mxtmin, (100))
        attributes[:, :, 4*  b + i] = attribute_area_filter(mxtmin, (200))
        attributes[:, :, 5 * b + i] = attribute_area_filter(mxtmin, (300))
    approfile= np.concatenate((attributes,imarray),axis=2)
    return approfile

def extinc_profile_single(imsingle):
    
    Bc = np.zeros((3,3),dtype = bool)
    Bc[1,:] = True
    Bc[:,1] = True
    
    # Parameters used to compute the extinction profile
    nextrema =  [int(2**jj) for jj in range(15)][::-1]
    
    # Array to store the profile
    H,W = imsingle.shape
    Z = 2*len(nextrema)+1
    ep = np.zeros((H,W,Z))
    #Min tree profile
    #Negating the image
    max_value =imsingle.max()
    data_neg = (max_value - imsingle)
 

    # Building the max-tree of the negated image, i.e. min-tree
    mxt = siamxt.MaxTreeAlpha(data_neg,Bc)

    # Area attribute extraction and computation of area extinction values
    area = mxt.node_array[3,:]
    ext = mxt.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxt.computeHeight()
    #ext = mxt.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    #volume = mxt.computeVolume()
    #ext = mxt.computeExtinctionValues(volume,"volume")
    
    # Min-tree profile
    i = len(nextrema) - 1
    for n in nextrema:
      mxt2 = mxt.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,i] = max_value - mxt2.getImage()
      i-=1
    i = len(nextrema)
    ep[:,:,i] = imsingle
    i +=1
    #Building the max-tree
    mxtmax = siamxt.MaxTreeAlpha(imsingle,Bc)

    # Area attribute extraction and computation of area extinction values
    area = mxtmax.node_array[3,:]
    ext = mxtmax.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxtmax.computeHeight()
    #ext = mxtmax.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    #volume = mxtmax.computeVolume()
    #Vext = mxtmax.computeExtinctionValues(volume,"volume")
    
    # Max-tree profile
    for n in nextrema:
      mxt2 = mxtmax.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,i] = mxt2.getImage()
      i+=1
    # Putting the original image in the profile    
    #image_array= np.concatenate((ep,imarray),axis=2)
    return ep
def extinc_profile_3D(imsingle):
    
    Bc = np.ones((3,3,3),dtype = bool)
    
    # Parameters used to compute the extinction profile
    nextrema =  [int(2**jj) for jj in range(15)][::-1]
    
    # Array to store the profile
    H,W,b = imsingle.shape
    Z = 2*b*len(nextrema)
    ep = np.zeros((H,W,Z))
    #Min tree profile
    #Negating the image
    max_value =imsingle.max()
    data_neg = (max_value - imsingle)
 

    # Building the max-tree of the negated image, i.e. min-tree
    mxt = siamxt.MaxTreeAlpha(data_neg,Bc)

    # Area attribute extraction and computation of area extinction values
    #area = mxt.node_array[3,:]
    #ext = mxt.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxt.computeHeight()
    #ext = mxt.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    volume = mxt.computeVolume()
    ext = mxt.computeExtinctionValues(volume,"volume")
    
    i = 0
    # Min-tree profile
    for n in nextrema:
      mxt2 = mxt.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,0+i:b+i] = max_value - mxt2.getImage()
      i+=b
    #Building the max-tree
    mxtmax = siamxt.MaxTreeAlpha(imsingle,Bc)

    # Area attribute extraction and computation of area extinction values
    #area = mxtmax.node_array[3,:]
    #ext = mxtmax.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxtmax.computeHeight()
    #Hext = mxtmax.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    volume = mxtmax.computeVolume()
    ext = mxtmax.computeExtinctionValues(volume,"volume")
    
    i = 0
    # Max-tree profile
    for n in nextrema:
      mxt2 = mxtmax.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,b*15+i:b*16+i]= mxt2.getImage()
      i+=b
    # Putting the original image in the profile    
    #image_array= np.concatenate((ep,imarray),axis=2)
    return ep
def extinc_profile_marginal(imarray):
    
    first=extinc_profile_single(imarray[:,:,0])
    sec=extinc_profile_single(imarray[:,:,1])
    third=extinc_profile_single(imarray[:,:,2])
    forth=extinc_profile_single(imarray[:,:,3])
    fifth=extinc_profile_single(imarray[:,:,4])
    sixth=extinc_profile_single(imarray[:,:,5])
    
    result=input= np.concatenate((first,sec,third,forth, fifth,sixth),axis=2)
 
    return result