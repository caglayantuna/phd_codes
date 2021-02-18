#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:14:04 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

def reference_tide(im):
    
    Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/reference0601.tif')
    imarray1=geoImToArray(Image1)

    Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/reference0701.tif')
    imarray2=geoImToArray(Image2)

    Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/reference0711.tif')
    imarray3=geoImToArray(Image3)

    Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/reference0731.tif')
    imarray4=geoImToArray(Image4)

    Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/reference0805.tif')
    imarray5=geoImToArray(Image5)

    imarray=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5))
    gt=imarray.max(axis=2)-imarray.min(axis=2)
    im[im==0]=1
    im=np.float64(im)
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    Precision=(TP)/(TP+FP )
 
    Recall=(TP)/(TP+FN )
 
    F1=(2*Precision*Recall)/(Precision+Recall)
    return F1

#water index
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180601ndwi.tif')
imarray1=geoImToArray(Image1)

Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180621ndwi.tif')
imarray2=geoImToArray(Image2)

Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180701ndwi.tif')
imarray3=geoImToArray(Image3)

Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180711ndwi.tif')
imarray4=geoImToArray(Image4)

Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180731ndwi.tif')
imarray5=geoImToArray(Image5)

Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180805ndwi.tif')
imarray6=geoImToArray(Image6)

imarray=np.dstack((imarray1,imarray3,imarray4,imarray5,imarray6))


stdimage=stdSITS(imarray)

stdimage[stdimage>50]=255


stdimage[stdimage<=50]=0


im_show(stdimage)
F1=reference_tide(stdimage)