#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:17:55 2020

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


def sth(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    
    areamax=mxt.node_array[3,:]
    areamin=mxtmin.node_array[3,:]
    
    dur=time_duration(mxt)
    durmin=time_duration(mxtmin)
    
    height=mxt.computeHeight()
    heightmin=mxtmin.computeHeight()

    np.savetxt('st_areamaxbrittany.txt',areamax,fmt='%d')
    np.savetxt('st_areasminbrittany.txt',areamin,fmt='%d')
    np.savetxt('st_durmaxbrittany.txt',dur,fmt='%d')
    np.savetxt('st_durminbrittany.txt',durmin,fmt='%d')   
    np.savetxt('st_heightmaxbrittany.txt',height,fmt='%d')
    np.savetxt('st_heightsminbrittany.txt',heightmin,fmt='%d')
    

def th(im):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.zeros((3, 3), dtype=bool)
    Bc[1,:]=1
    Bc[:,1]=1
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mxtmin=siamxt.MaxTreeAlpha(imarraymin[:, :, i], Bc)
        areamax=mxt.node_array[3,:]
        areamin=mxtmin.node_array[3,:]
    
        height=mxt.computeHeight()
        heightmin=mxtmin.computeHeight()
 
        np.savetxt('th_areamaxbrittany'+str(i)+'.txt',areamax,fmt='%d')
        np.savetxt('th_areasminbrittany'+str(i)+'.txt',areamin,fmt='%d')   
        np.savetxt('th_heightmaxbrittany'+str(i)+'.txt',height,fmt='%d')
        np.savetxt('th_heightsminbrittany'+str(i)+'.txt',heightmin,fmt='%d')




Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_int.tif') 
#Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif') 


imarray = geoImToArray(Image)
sth(imarray)
th(imarray)