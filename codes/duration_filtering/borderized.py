#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:03:56 2020

@author: caglayantuna
"""

from functions.project_functions import *
from imageio import imsave
from imageio import imread

def border(im):
    im[0,:]=0
    im[-1,:]=0
    im[:,-1]=0
    im[:,0]=0
    return im
    
    
#data pleiades   
Image1 = imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detectedreal1.png')
Image2=imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detectedreal2.png')
Image3 = imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detectedreal3.png')
Image4 = imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detectedreal4.png')
Image5 =imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detectedreal5.png')
Image6 =imread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/detecetdreal6.png')

border1=border(Image1)[:,:,:3]
border2=border(Image2)[:,:,:3]
border3=border(Image3)[:,:,:3]
border4=border(Image4)[:,:,:3]
border5=border(Image5)[:,:,:3]
border6=border(Image6)[:,:,:3]


imsave('detectedreal1.png',border1)
imsave('detectedreal2.png',border2)
imsave('detectedreal3.png',border3)
imsave('detectedreal4.png',border4)
imsave('detectedreal5.png',border5)
imsave('detecetdreal6.png',border6)




