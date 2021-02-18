#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:41:19 2019

@author: caglayantuna
"""
import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
from skimage import exposure
def accuracy(im):
    #Image = geoimread('grountruhrasterized.png')
    
    Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    im=np.float64(im)
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
#first dataset   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3= geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#second dataset   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata2.png')
Image3= geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata3.png')

imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
#imarray4=data_prepare(Image4)



imarray=np.dstack([imarray1,imarray2,imarray3])

imarray=255*(imarray/imarray.max())
imarray=np.array(imarray,dtype=np.uint8)


#min tree
Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True
tree1 = siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
#tree1.areaOpen(200)
c=duration_filter(tree1,1)#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree1)



nodes=np.where(tempstabil<0.5)[0]
#nodes=nodes[6:]
result=sum_of_nodes(tree1,nodes)
node_show(result)

flood=result[:,:,0]-result[:,:,1]
flood[flood>20]=255
bc = np.ones((3,3), dtype = bool)

tree2 = siamxt.MaxTreeAlpha(flood, bc)
flood=attribute_area_filter(tree2,50)
TP, TN, FP, FN=accuracy(flood)

print((TP, TN, FP, FN )) 

#leveldiff=parent_level_difference(tree)
#areadiff=parent_area_difference(tree)

#multiply=np.multiply(areadiff,leveldiff)

Precision=(TP)/(TP+FP )
 
Recall=(TP)/(TP+FN )
 
F1=(2*Precision*Recall)/(Precision+Recall)