#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:39:09 2020

@author: caglayantuna
"""

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
from imageio import imwrite
from scipy.misc import imread

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

imgeo= geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180805morbihanclip.tif')

deneme=np.copy(imarray)

deneme[deneme>120]=255
deneme[deneme<=120]=0

node_show(imarray)
node_show(deneme)

#for i in range(deneme.shape[2]):
    
 #    imwrite('tresholdedndwi'+str(i)+".tif",deneme[:,:,i])

#for i in range(deneme.shape[2]):
    
 #   imwrite('tresholdedndwi'+str(i)+".tif",deneme[:,:,i])
 
 
#array_to_raster(imarray1,imgeo,'20180601ndwi.tif')
#array_to_raster(imarray3,imgeo,'20180701ndwi.tif')
#array_to_raster(imarray4,imgeo,'20180711ndwi.tif')
#array_to_raster(imarray5,imgeo,'20180731ndwi.tif')
#array_to_raster(imarray6,imgeo,'20180805ndwi.tif')


#a=imread('reference0601.png')
#array_to_raster(a,imgeo,'reference0601.tif')
#a=imread('reference0701.png')
#array_to_raster(a,imgeo,'reference0701.tif')
#a=imread('reference0711.png')
#array_to_raster(a,imgeo,'reference0711.tif')
#a=imread('reference0731.png')
#array_to_raster(a,imgeo,'reference0731.tif')
#a=imread('reference0805.png')
#array_to_raster(a,imgeo,'reference0805.tif')