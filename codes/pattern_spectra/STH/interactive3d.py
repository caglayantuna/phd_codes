#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:39:10 2019

@author: caglayantuna
"""
import siamxt
from functions.project_functions import *
from scipy import stats

Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/toulouse/2018ndvimerged.tif')
#Image1 = geoimread('psdata/morbihanndvi.tif')

imarray1=geoImToArray(Image1)
imarray1=np.array(imarray1,dtype=np.uint16)


#Image2 = geoimread('psdata/secondps.tif')
#Image2 = geoimread('psdata/2017ndvimerged.tif')

#imarray2=geoImToArray(Image2)
#imarray2=np.array(imarray2,dtype=np.uint16)



#tree building
#Bc = np.ones((3,3,3), dtype=bool)
Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True
#max tree
#tree1 = siamxt.MaxTreeAlpha(imarray1, Bc)
#tree2 = siamxt.MaxTreeAlpha(imarray2, Bc)
#min tree
tree1 = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)
#tree1.node_array[2,:]=imarray1.max()-tree1.node_array[2,:]
#tree2 = siamxt.MaxTreeAlpha(imarray2.max()-imarray2, Bc)

#area weighted parent level difference
ps1=area_weighted(tree1)
#ps1vector=np.reshape(ps1,imarray1.size)
#ps2=area_weighted(tree2)
#ps2vector=np.reshape(ps2,[30*31*6])

#area attribute
area1=area_vector(tree1)
#area2=area_vector(tree2)

#shape attribute
#shape1=eccentricity_vector(tree1)
#shape2=eccentricity_vector(tree2)

shape1=tree1.computeRR()
#shape1=mean_vector(tree1)
#shape2=tree2.computeRR()

#shape1=tree1.node_array[12,:]
#shape2=tree2.node_array[12,:]
#shape1=tree1.computeNodeGrayAvg()
#shape2=tree2.computeNodeGrayAvg()

#time attribute
time1=time_end(tree1)
#time2=timevector(tree2)

#time1=tree1.computeNodeCentroid()
#time1=time1[:,2]
#time1=np.sum(time1**2, axis=1)
#time2=tree2.computeNodeCentroid()
#time2=np.sum(time2**2, axis=1)
#time2=time2[:,2]




#3D PATTERN SPECTRA
data1=np.column_stack((area1,shape1,time1)) 
#data2=np.column_stack((area2,shape2,time2)) 

statistic1,edges1,binnumber1=stats.binned_statistic_dd(data1,ps1,statistic='sum', bins=[100,100,7])
#statistic2,edges2,binbumber2=stats.binned_statistic_dd(data2,ps2,statistic='sum', bins=[100,100,7])

a1=pattern_spectra_nodes3d(tree1,edges1[0],edges1[1],edges1[2],1,27,6,area1, shape1,time1)#area ecc
a2=pattern_spectra_nodes3d(tree1,edges1[0],edges1[1],edges1[2],0,26,2,area1, shape1,time1)#area ecc