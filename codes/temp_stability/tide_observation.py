#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:21:56 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
from skimage import exposure

def simplify_tree(tree):
    #simplify tree
    nodes=np.where(tree.node_array[1,:]>1)[0]
    nodeschildren=[]
    for i in range(nodes.shape[0]):
       children=tree.getChildren(nodes[i])
       nodeschildren.append(children)
    nodeschildren = np.array([ elem for singleList in nodeschildren for elem in singleList])
    nodespruned=np.ones(tree.node_array.shape[1])
    nodespruned[nodeschildren]=False
    nodespruned[0]=False #root
    nodespruned=np.array(nodespruned,dtype=int)
    #nodespruned=np.array(nodespruned, dtype=bool)
    newtree=tree.clone()
    a=newtree.prune(nodespruned)
    return a
def reference_tide(output):
    
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
    TP=np.zeros((5))
    FP=np.zeros((5))
    FN=np.zeros((5))
    TN=np.zeros((5))
    r,c,b=output.shape
    for i in range(b):
            im=output[:,:,i]
            gt=imarray[:,:,i]
            im[im>0]=255    
            im[im==0]=1
            im=np.float64(im)
            gt=np.float64(gt)
            res=im-gt
            TP[i]=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
            TN[i]=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
            FP[i]=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
            FN[i]=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    TP=np.mean(TP)
    TN=np.mean(TN)
    FP=np.mean(FP)
    FN=np.mean(FN)
    Precision=(TP)/(TP+FP )
 
    Recall=(TP)/(TP+FN )
 
    F1=(2*Precision*Recall)/(Precision+Recall)       
    return F1
    
    
def plot_result(im,res):
    fig=plt.figure()
    plt.subplot(2, 6, 1)
    plt.imshow(im[:,:,0], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 2)
    plt.imshow(im[:,:,1], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 3)
    plt.imshow(im[:,:,2], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 4)
    plt.imshow(im[:,:,3], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 5)
    plt.imshow(im[:,:,4], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 6)
    plt.imshow(im[:,:,5], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 7)
    plt.imshow(res[:,:,0], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 8)
    plt.imshow(res[:,:,1], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 9)
    plt.imshow(res[:,:,2], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 10)
    plt.imshow(res[:,:,3], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 11)
    plt.imshow(res[:,:,4], cmap='gray')
    plt.title('')
    plt.subplot(2, 6, 12)
    plt.imshow(res[:,:,5], cmap='gray')
    plt.title('')
    fig.set_size_inches(50, fig.get_figheight(), forward=True)
    plt.show()
#Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
#imarray=geoImToArray(Image)
#imarray=np.array(imarray,dtype=np.uint16)

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
#tree building
#Bc = np.ones((3,3,3), dtype=bool)
Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

#hyperconnectivity
#tree= siamxt.MaxTreeAlpha(imarray, Bc)
#treenew=simplify_tree(tree)
#imarray=treenew.getImage()

#min tree
tree1 = siamxt.MaxTreeAlpha(imarray, Bc)

c=duration_filter(tree1,3)
tree2 = siamxt.MaxTreeAlpha(c, Bc)

c=attribute_area_filter(tree2,300)
#tempstabil=temp_stability_range(tree1)
tempstabil=temp_stability_ratio(tree2)



nodes=np.where(tempstabil<0.7)[0]
#nodes=nodes[67:]
result=sum_of_nodes(tree2,nodes)
#tree2 = siamxt.MaxTreeAlpha(result, Bc)
#result2=attribute_area_filter(tree2,300)
node_show(result)

F1=reference_tide(result)
#plot_result(imarray,result)