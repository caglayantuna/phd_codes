#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:54:10 2020

@author: caglayantuna
"""

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
    #res=np.zeros([r,c,b])
    res=np.copy(im)
    for i in range(b):
        coord=np.where(ident[:,:,i]!=0)
        coord=np.array(coord)
        for j in range(coord.shape[1]):
            #x = np.arange(0, b)
            #c=interp1d(x,im[coord[0,j],coord[1,j],:])
            #xnew = np.arange(0, 6, 1)
            #res[coord[0,j],coord[1,j],:]=c(xnew)
            #array=np.delete(im[coord[0,j],coord[1,j],:],i)
            res[coord[0,j],coord[1,j],i]=np.mean(im[coord[0,j],coord[1,j],:])
    return res
def rmse_for_all(im,res):  
    gt=np.copy(im)
    r,c,b=gt.shape
    result=np.zeros(b)
    for i in range(b):
        result[i]=np.sqrt(mean_squared_error(gt[:,:,i], res[:,:,i]))
    return result
def road_extraction(im):
    res=np.zeros(im.shape)
    res[im<105]=255
    res[im>160]=255
    return res
def accuracy(im):
    r,c=im.shape
    gt=np.zeros([r,c])
    gt[35:41,254:264]=255
    gt[130:140,62:68]=255
    gt[220:230,62:68]=255
    gt[173:179,183:193]=255
    gt[180:186,180:190]=255
    gt[180:186,13:23]=255
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
def back_sub(gt):  
    r,c,b=gt.shape
    gt=np.array(gt,np.float)
    result=np.zeros([r,c],dtype=np.float)
    for i in range(b-1):
        result+=gt[:,:,i]-gt[:,:,i+1]   
    return np.abs(result)
def filter_per_single(merged,identified,t):
    im=merged[:,:,t]
    bc = np.ones((3,3), dtype=bool) 
    tree = siamxt.MaxTreeAlpha(im, bc)
    coord=np.where(identified[:,:,t]>0)
    nodespruned=tree.node_index[coord]
    nodespruned=np.unique(nodespruned)
    nodes=np.zeros(tree.node_array.shape[1])
    nodes[nodespruned]=True
    nodes=np.array(nodes,dtype=int)
    nodes=np.array(nodes, dtype=bool)
    tree.prune(nodes)  
    res=tree.getImage()
    return res
    
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
identified=merged-filtered

res1=filter_per_single(merged,identified,0)
res2=filter_per_single(merged,identified,1)
res3=filter_per_single(merged,identified,2)
res4=filter_per_single(merged,identified,3)
res5=filter_per_single(merged,identified,4)
res6=filter_per_single(merged,identified,5)


result=np.dstack((res1,res2,res3,res4,res5,res6))
#node_show(filtered)
#node_show(result)

rmse_tree=rmse_for_all(ref,result)
