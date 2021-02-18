#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:58:24 2020

@author: caglayantuna
"""

from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.insert(0,path)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions.project_functions import *
import siamxt 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imageio import imwrite


def analysis(im,x):
    c=np.copy(imarray)

    c[c>1]=x
    c[c<-1]=-x
    c=c*255
    node_show(c[2000:2500,2000:2500,:])

Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_imposed.tif') 

imarray = NDVI_to_array(Image)



#first
analysis(imarray,1)



#second
analysis(imarray,0)