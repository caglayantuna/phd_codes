#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:02:33 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
def plot_result(im,res):
    fig=plt.figure()
    plt.subplot(2, 6, 1)
    plt.imshow(im[:,:,0], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 2)
    plt.imshow(im[:,:,1], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 3)
    plt.imshow(im[:,:,2], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 4)
    plt.imshow(im[:,:,3], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 5)
    plt.imshow(im[:,:,4], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 6)
    plt.imshow(im[:,:,5], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 7)
    plt.imshow(res[:,:,0], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 8)
    plt.imshow(res[:,:,1], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 9)
    plt.imshow(res[:,:,2], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 10)
    plt.imshow(res[:,:,3], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 11)
    plt.imshow(res[:,:,4], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    plt.subplot(2, 6, 12)
    plt.imshow(res[:,:,5], cmap='gray',vmin=0,vmax=255)
    plt.title('')
    fig.set_size_inches(30, fig.get_figheight(), forward=True)
    plt.show()

Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')

imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint16)
#tree building
Bc = np.ones((3,3,3), dtype = bool)

node_show(imarray)
#min tree
tree = siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)

#max tree
treemax= siamxt.MaxTreeAlpha(imarray, Bc)


time=time_duration(treemax)
treemax.node_array[3,:]=time
filtered=attribute_area_filter(treemax,4)
node_show(filtered)


#node_show(10*(imarray-filteredmin))
#imsave("dur1.png",result[:,:,0])
#imsave("dur2.png",result[:,:,1])
#imsave("dur3.png",result[:,:,2])
#imsave("dur4.png",result[:,:,3])
#imsave("dur5.png",result[:,:,4])
#imsave("dur6.png",result[:,:,5])