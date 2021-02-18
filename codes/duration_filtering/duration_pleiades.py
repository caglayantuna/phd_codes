#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:46:46 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from imageio import imsave

def road_extraction(im):
    res=np.zeros(im.shape)
    res[im<105]=255
    res[im>175]=255
    return res
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

def colorized(im1,im2,d,t):
    r,c,b=im1.shape
    a=np.zeros([r,c,3])
    a[im1[:,:,d]==t,:]=[0,255,0]
    a[im2[:,:,d]==t,:]=[255,0,0]
    a=np.uint8(a)
    return a

def duration_image(tree):
    duration=time_duration(tree)
    dur_img=duration[tree.node_index]
    return dur_img
def filter_colorized(im):
    r,c,b=im.shape
    filtered=np.zeros([r,c,b])
    for i in range(b):
       Bc = np.ones((3,3), dtype=bool) 
       treemax = siamxt.MaxTreeAlpha(im[:,:,i], Bc)
       filtered[:,:,i]=attribute_area_filter(treemax,10)
       filtered[:,:,i]=filtered[:,:,i]
       #filtered[:,:,i][filtered[:,:,i] == 0]=255
    d=np.sum(filtered,axis=2)
    filtered[d==0]=[255,255,255] 
    filtered[0,:,:]=[0,0,0]
    filtered[-1,:,:]=[0,0,0]
    filtered[:,-1,:]=[0,0,0]
    filtered[:,0,:]=[0,0,0]

    filtered=np.uint8(filtered)
    return filtered
    
       
#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew6.png')

#data pleiades   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')
#Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall4.png')
#Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall5.png')
#Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall6.png')


#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)
imarray6=geoImToArray(Image6)
merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)

#min tree
Bc = np.ones((3,3,3), dtype=bool) 
tree = siamxt.MaxTreeAlpha(merged.max()-merged, Bc)
tree.node_array[2,:]=merged.max()-tree.node_array[2,:]

#filtering
#filtered1=duration_filter(treemax,0)[170:230,80:150]
#filtered2=duration_filter(treemax,1)[170:230,80:150]
#filtered3=duration_filter(treemax,2)[170:230,80:150]
#filtered4=duration_filter(treemax,3)[170:230,80:150]
#filtered5=duration_filter(treemax,4)[170:230,80:150]
#filtered6=duration_filter(treemax,5)[170:230,80:150]

#filtering
#filteredmin1=duration_filter(tree,0)[170:230,80:150]
#filteredmin2=duration_filter(tree,1)[170:230,80:150]
#filteredmin3=duration_filter(tree,2)[170:230,80:150]
#filteredmin4=duration_filter(tree,3)[170:230,80:150]
#filteredmin5=duration_filter(tree,4)[170:230,80:150]
#filteredmin6=duration_filter(tree,5)[170:230,80:150]


#identified1=merged[170:230,80:150,:]-filtered1+merged[170:230,80:150,:]-filteredmin1
#identified2=merged[170:230,80:150,:]-filtered2+merged[170:230,80:150,:]-filteredmin2
#identified3=merged[170:230,80:150,:]-filtered3+merged[170:230,80:150,:]-filteredmin3
#identified4=merged[170:230,80:150,:]-filtered4+merged[170:230,80:150,:]-filteredmin4
#identified5=merged[170:230,80:150,:]-filtered5+merged[170:230,80:150,:]-filteredmin5
#identified6=merged[170:230,80:150,:]-filtered6+merged[170:230,80:150,:]-filteredmin6


durmax=duration_image(treemax)
durmin=duration_image(tree)

deneme=colorized(durmax,durmin,0,0)[170:230,140:180,:]

#im_show(deneme)

#node_show(filteredmin)
#node_show(merged-filtered)
#node_show(filteredmin-merged)
#node_show(detect_moving_nodes(treemax))


#im_show(road_extraction(merged[:,:,3]))
#im_show(road_extraction(filtered[:,:,3]))


imsave('results/real_SITS1.png',merged[170:230,140:180,0])
imsave('results/real_SITS2.png',merged[170:230,140:180,1])
imsave('results/real_SITS3.png',merged[170:230,140:180,2])
imsave('results/real_SITS4.png',merged[170:230,140:180,3])
imsave('results/real_SITS5.png',merged[170:230,140:180,4])
imsave('results/real_SITS6.png',merged[170:230,140:180,5])

imsave('results/duration1real_SITS1area.png',filter_colorized(colorized(durmax,durmin,0,0)[170:230,140:180,:]))
imsave('results/duration1real_SITS2area.png',filter_colorized(colorized(durmax,durmin,1,0)[170:230,140:180,:]))
imsave('results/duration1real_SITS3area.png',filter_colorized(colorized(durmax,durmin,2,0)[170:230,140:180,:]))
imsave('results/duration1real_SITS4area.png',filter_colorized(colorized(durmax,durmin,3,0)[170:230,140:180,:]))
imsave('results/duration1real_SITS5area.png',filter_colorized(colorized(durmax,durmin,4,0)[170:230,140:180,:]))
imsave('results/duration1real_SITS6area.png',filter_colorized(colorized(durmax,durmin,5,0)[170:230,140:180,:]))

imsave('results/duration2real_SITS1.png',filter_colorized(colorized(durmax,durmin,0,1)[170:230,140:180,:]))
imsave('results/duration2real_SITS2.png',filter_colorized(colorized(durmax,durmin,1,1)[170:230,140:180,:]))
imsave('results/duration2real_SITS3.png',filter_colorized(colorized(durmax,durmin,2,1)[170:230,140:180,:]))
imsave('results/duration2real_SITS4.png',filter_colorized(colorized(durmax,durmin,3,1)[170:230,140:180,:]))
imsave('results/duration2real_SITS5.png',filter_colorized(colorized(durmax,durmin,4,1)[170:230,140:180,:]))
imsave('results/duration2real_SITS6.png',filter_colorized(colorized(durmax,durmin,5,1)[170:230,140:180,:]))

imsave('results/duration3real_SITS1.png',filter_colorized(colorized(durmax,durmin,0,2)[170:230,140:180,:]))
imsave('results/duration3real_SITS2.png',filter_colorized(colorized(durmax,durmin,1,2)[170:230,140:180,:]))
imsave('results/duration3real_SITS3.png',filter_colorized(colorized(durmax,durmin,2,2)[170:230,140:180,:]))
imsave('results/duration3real_SITS4.png',filter_colorized(colorized(durmax,durmin,3,2)[170:230,140:180,:]))
imsave('results/duration3real_SITS5.png',filter_colorized(colorized(durmax,durmin,4,2)[170:230,140:180,:]))
imsave('results/duration3real_SITS6.png',filter_colorized(colorized(durmax,durmin,5,2)[170:230,140:180,:]))

imsave('results/duration4real_SITS1.png',filter_colorized(colorized(durmax,durmin,0,3)[170:230,140:180,:]))
imsave('results/duration4real_SITS2.png',filter_colorized(colorized(durmax,durmin,1,3)[170:230,140:180,:]))
imsave('results/duration4real_SITS3.png',filter_colorized(colorized(durmax,durmin,2,3)[170:230,140:180,:]))
imsave('results/duration4real_SITS4.png',filter_colorized(colorized(durmax,durmin,3,3)[170:230,140:180,:]))
imsave('results/duration4real_SITS5.png',filter_colorized(colorized(durmax,durmin,4,3)[170:230,140:180,:]))
imsave('results/duration4real_SITS6.png',filter_colorized(colorized(durmax,durmin,5,3)[170:230,140:180,:]))

imsave('results/duration5real_SITS1.png',filter_colorized(colorized(durmax,durmin,0,4)[170:230,140:180,:]))
imsave('results/duration5real_SITS2.png',filter_colorized(colorized(durmax,durmin,1,4)[170:230,140:180,:]))
imsave('results/duration5real_SITS3.png',filter_colorized(colorized(durmax,durmin,2,4)[170:230,140:180,:]))
imsave('results/duration5real_SITS4.png',filter_colorized(colorized(durmax,durmin,3,4)[170:230,140:180,:]))
imsave('results/duration5real_SITS5.png',filter_colorized(colorized(durmax,durmin,4,4)[170:230,140:180,:]))
imsave('results/duration5real_SITS6.png',filter_colorized(colorized(durmax,durmin,5,4)[170:230,140:180,:]))

