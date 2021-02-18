#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:53:05 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave

def normalize_im(im):
   im=(im/im.max())*255
   im=np.array(im,dtype=np.uint8)
   return im
def detect_moving_nodes(tree):
    nodes=np.where(tree.node_array[13,:]-tree.node_array[12,:]==0)
    nodespruned=np.ones(tree.node_array.shape[1])
    nodespruned[nodes]=False
    nodespruned[0]=False #root 
    nodespruned=np.array(nodespruned, dtype=bool)

    tree.prune(nodespruned)
    nodespruned=np.array(nodespruned, dtype=bool)
    image=tree.getImage()
    return image   

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
    fig.set_size_inches(30, fig.get_figheight(), forward=True)
    plt.show()
def duration_filter_min(imarray,Bc):
    tree = siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)
    #duration filktering with min
    time=time_duration(tree)
    tree.node_array[2,:]=imarray.max()-tree.node_array[2,:]
    tree.node_array[3,:]=time
    filteredmin=attribute_area_filter(tree,0)
    #plot_result(30*imarray,30*filteredmin)
    return filteredmin

def duration_filter_max(imarray,Bc):
    tree = siamxt.MaxTreeAlpha(imarray, Bc)
    time=time_duration(tree)
    tree.node_array[3,:]=time
    filteredmax=attribute_area_filter(tree,0)
    #plot_result(30*imarray,30*filteredmax)
    return filteredmax
def show_colored(t):
    result=np.zeros(imarray1.shape,dtype=np.uint8)

    result[:,:,0]=filtered1[:,:,t]
    result[:,:,1]=filtered2[:,:,t]
    result[:,:,2]=filtered3[:,:,t]
    im_show(2*result[:,:,1:4])
    
    
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180601morbihanclip.tif')
imarray1=geoImToArray(Image1)

Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180621morbihanclip.tif')
imarray2=geoImToArray(Image2)

Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180701morbihanclip.tif')
imarray3=geoImToArray(Image3)

Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180706morbihanclip.tif')
imarray4=geoImToArray(Image4)

Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180711morbihanclip.tif')
imarray5=geoImToArray(Image5)

Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180731morbihanclip.tif')
imarray6=geoImToArray(Image6)

Image7 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/20180805morbihanclip.tif')
imarray7=geoImToArray(Image7)

allfirstband=np.dstack((imarray1[:,:,0],imarray2[:,:,0],imarray3[:,:,0],imarray4[:,:,0],imarray5[:,:,0],imarray6[:,:,0],imarray7[:,:,0]))
allfirstband=normalize_im(allfirstband)

allsecondband=np.dstack((imarray1[:,:,1],imarray2[:,:,1],imarray3[:,:,1],imarray4[:,:,1],imarray5[:,:,1],imarray6[:,:,1],imarray7[:,:,1]))
allsecondband=normalize_im(allsecondband)


allthirdband=np.dstack((imarray1[:,:,2],imarray2[:,:,2],imarray3[:,:,2],imarray4[:,:,2],imarray5[:,:,2],imarray6[:,:,2],imarray7[:,:,2]))
allthirdband=normalize_im(allthirdband)


allforthband=np.dstack((imarray1[:,:,3],imarray2[:,:,3],imarray3[:,:,3],imarray4[:,:,3],imarray5[:,:,3],imarray6[:,:,3],imarray7[:,:,3]))
allforthband=normalize_im(allforthband)


Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True


filtered1=duration_filter_min(allfirstband,Bc)
filtered2=duration_filter_min(allsecondband,Bc)
filtered3=duration_filter_min(allthirdband,Bc)
filtered4=duration_filter_min(allforthband,Bc)

filtered1=duration_filter_max(allfirstband,Bc)
filtered2=duration_filter_max(allsecondband,Bc)
filtered3=duration_filter_max(allthirdband,Bc)
filtered4=duration_filter_max(allforthband,Bc)

node_show(allforthband)
#node_show(filtered1)
#node_show(filtered2)
#node_show(filtered3)
node_show(filtered4)
treemax = siamxt.MaxTreeAlpha(allforthband, Bc)
movingnodes=detect_moving_nodes(treemax)
node_show(movingnodes)
#im_show(normalize_im(imarray1[:,:,1:4]))
#im_show(normalize_im(imarray2[:,:,1:4]))
#im_show(normalize_im(imarray3[:,:,1:4]))
#im_show(normalize_im(imarray4[:,:,1:4]))
#im_show(normalize_im(imarray5[:,:,1:4]))
#im_show(normalize_im(imarray6[:,:,1:4]))
#im_show(normalize_im(imarray7[:,:,1:4]))


#reconstruct filtered 
result=np.zeros(imarray1.shape,dtype=np.uint8)
result[:,:,0]=filtered1[:,:,2]
result[:,:,1]=filtered2[:,:,2]
result[:,:,2]=filtered3[:,:,2]

#show colorse
#show_colored(0)
#show_colored(1)
#show_colored(2)
#show_colored(3)
#show_colored(4)
#show_colored(5)
#show_colored(6)