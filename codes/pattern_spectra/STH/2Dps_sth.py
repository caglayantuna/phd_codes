#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:17:33 2019

@author: caglayantuna
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:14:35 2019

@author: caglayantuna
"""
import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave


#synthetic data 
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd1.png')
#imarray1=geoImToArray(Image1)[:,:,3]
#imarray1= imarray1.astype(np.uint8)

#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd2.png')
#imarray2=geoImToArray(Image2)[:,:,3]
#imarray2= imarray2.astype(np.uint8)

#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd3.png')
#imarray3=geoImToArray(Image3)[:,:,3]
#imarray3= imarray3.astype(np.uint8)


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall5.png')


#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)


merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
tree1 = siamxt.MaxTreeAlpha(merged, Bc)


#mIN tree
#Bc = np.ones((3,3,3), dtype=bool) 
#tree1 = siamxt.MaxTreeAlpha(merged.max()-merged, Bc)
#tree1.node_array[3,:]=merged.max()-tree1.node_array[2,:]

#area weighted parent level difference
ps1=area_weighted(tree1)
#ps1vector=np.reshape(ps1,imarray1.size)
#ps2vector=np.reshape(ps2,[30*31*6])

#area attribute
#area1=mean_vector(tree1)
area1=area_vector(tree1)
#shape attribute
#shape1=eccentricity_vector(tree1)
#area1=tree1.computeVolume()
#shape1=mean_vector(tree1)
#shape1=tree1.computeStabilityMeasure()

#time attribute
time1=temp_stability_ratio(tree1)
#time1=tree1.computeNodeCentroid()[:,2]
#time1=time_duration(tree1)

#3D PATTERN SPECTRA
#data1=np.column_stack((area1,shape1,time1)) 
#data2=np.column_stack((area2,shape2,time2)) 

res11,binedges11,binedges12,res=stats.binned_statistic_2d(area1,time1,ps1,statistic='sum', bins=[50,50])
res11 = res11.astype(np.uint16)
#res21 = res21.astype(np.uint16)
#res31 = res31.astype(np.uint16)



#pattern spectra space
bc = np.zeros((3,3), dtype = bool)
bc[1,:] = True
bc[:,1] = True
treelast = siamxt.MaxTreeAlpha(res11, bc)
filtered=attribute_area_filter(treelast,10)
#treelast2 = siamxt.MaxTreeAlpha(res21, Bc)
#filtered2=attribute_area_filter(treelast2,80)


#extract all bins first image
bins=np.nonzero(filtered)
result=np.zeros(tree1.shape)
bins=np.array(bins)


nodes=get_nodes_from_bins(binedges11,binedges12,bins,area1, time1)
result=reconstruct_nodes(tree1,nodes)

node_show(merged)

#max

node_show(result)
#min
#node_show(merged.max()-result)